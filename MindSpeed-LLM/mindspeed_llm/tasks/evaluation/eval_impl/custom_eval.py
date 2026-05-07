# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import math
import glob
import shutil
import logging
import json
import pandas as pd
import tqdm
import torch
from typing import Tuple, List, Dict, Any

from megatron.core import mpu
from megatron.training import get_args
from torch import distributed as dist
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.eval_utils.gsm8k_utils import four_shots_prompt, gsm8k_postprocess
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset
from mindspeed_llm.tasks.evaluation.eval_impl.template import GSM8K_TEMPLATE_DIR

logger = logging.getLogger(__name__)


class CustomEval(DatasetEval):
    def __init__(self, input_dir, eval_args):
        self.input_dir = input_dir
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.broadcast_rank = [[0]]
        self.max_eval_samples = eval_args.max_eval_samples  # 可选，仅对min(max_eval_samples, len(eval_data_list))进行评估
        self.task_type = ("general", "function_call", "jsonoutput", "nl2sql", "knowledge_graph_extraction")  # 当前支持任务类型
        # 初始化输出路径
        self.output_dir = eval_args.output_dir
        self.init_path(self.output_dir)

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        args = get_args()
        self.output_dir = args.output_dir
        # valid file format
        self.verify_data_format()

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=CustomEval.count_files(self.input_dir),
                                       desc="total eval files inference...")

        for root_dir, sub_dirs, files in os.walk(self.input_dir):
            output_subdir = self.init_output_path(root_dir)

            for file in files:
                file_path = os.path.join(root_dir, file)
                eval_data_list = self.read_jsonl_file(file_path)

                instructions = []
                ground_truths = []
                eval_data_list, group, align_start_dp_rank = self.get_final_data_list(args, eval_data_list)
                if self.rank == 0:
                    self.task_pbar = tqdm.tqdm(total=len(eval_data_list), desc=f"{file}推理中...")

                rank_inference_results = []  # save current rank inference result
                index = 0
                for _, item in enumerate(eval_data_list):
                    instructions.append(item)
                    ground_truths.append(item['data'][-1]['content'])

                    if len(instructions) == self.batch_size or len(eval_data_list) == index + 1:
                        chat_results, _ = chat.chat(instruction=instructions, history=[])
                        if 0 <= align_start_dp_rank <= mpu.get_data_parallel_rank() \
                                and len(eval_data_list) == index + 1:
                            chat_results = chat_results[:-1]
                        for idx, chat_result in enumerate(chat_results):
                            rank_inference_results.append(
                                {
                                    "instruction": instructions[idx],
                                    "inference_result": chat_result,
                                    "ground_truth": ground_truths[idx]
                                }
                            )
                            if dist.get_rank() in group[0]:
                                logger.info(f"correct: {ground_truths[idx]}, AI: {chat_result}, rank: {self.rank}")
                        instructions = []
                        ground_truths = []

                    if self.task_pbar is not None:
                        self.task_pbar.update()

                    index += 1

                if self.task_pbar is not None:
                    self.task_pbar.close()

                if self.file_pbar is not None:
                    self.file_pbar.update()

                # gather inference result from all device
                if self.rank not in group[0]:  # only rank in group[0] need to save
                    rank_inference_results = []
                gathered_results = self.gather_inference_result(rank_inference_results)

                # inference result save
                self.save_inference_file(gathered_results, output_subdir, file, group)

    def gather_inference_result(self, inference_results):
        gathered_results = [None] * dist.get_world_size() if self.rank == 0 else []
        dist.gather_object(
            obj=inference_results,
            dst=0,
            object_gather_list=gathered_results
        )
        return gathered_results

    def save_inference_file(self, inference_result, output_path, file_name, group):
        if self.rank == 0 and self.rank in group[0]:
            final_results = []
            for results_list in inference_result:
                final_results.extend(results_list)
            output_file_path = os.path.join(f"{output_path}", file_name.replace(".jsonl", f"_result.jsonl"))
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for result_item in final_results:
                    f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            print(f"origin file: {file_name}; infernece result save to: {output_file_path}")

    @staticmethod
    def validate_data_format(data: Dict[str, Any]):
        """针对待评估数据集做标准格式校验。标准格式参照OBP 26.1交付的四大模型特性格式"""
        # 1. 检查顶级键和类型
        if not isinstance(data, dict):
            raise ValueError("输入不是一个字典。")
        if "meta_prompt" not in data or "data" not in data:
            raise ValueError("缺少 'meta_prompt' 或 'data' 顶级键。")

        # 2. 检查 meta_prompt 字段
        meta_prompt = data["meta_prompt"]
        if not isinstance(meta_prompt, list) or not all(isinstance(item, str) for item in meta_prompt):
            raise ValueError("'meta_prompt' 字段必须是包含字符串的列表。")

        # 3. 检查 data 字段
        messages = data["data"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("'data' 字段必须是非空列表。")

        # 4. 检查 data 列表中的每个字典
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"'data' 列表中的第 {i} 个元素不是字典。")
            if "role" not in message or "content" not in message:
                raise ValueError(f"'data' 列表中第 {i} 个字典缺少 'role' 或 'content' 键。")
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                raise ValueError(f"'data' 列表中第 {i} 个字典的 'role' 或 'content' 字段必须是字符串。")

    @staticmethod
    def count_files(directory: str) -> int:
        """
        递归地统计一个目录及其所有子目录中的文件总数。
        """
        # 检查路径是否存在且是目录
        if not os.path.isdir(directory):
            print(f"错误: 目录 '{directory}' 不存在或不是一个有效的目录。")
            return 0

        file_count = 0

        for root, dirs, files in os.walk(directory):
            file_count += len(files)

        return file_count

    def sample_eval_data(self, eval_data_list, file_path):
        if self.max_eval_samples is not None:
            origin_len = len(eval_data_list)
            eval_data_list = (
                eval_data_list[0:min(self.max_eval_samples, origin_len)]
            )

            logger.info("%s length from %s to %s !!!", file_path, str(origin_len), str(len(eval_data_list)))
        return eval_data_list

    def init_path(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def init_output_path(self, root_dir):
        relative_path = os.path.relpath(root_dir, self.input_dir)
        output_subdir = os.path.join(self.output_dir, relative_path)
        if self.rank == 0:
            self.init_path(output_subdir)
        return output_subdir

    def read_jsonl_file(self, file_path) -> List:
        with open(file_path, encoding='utf-8') as f:
            eval_data_list = []
            for line in f.readlines():
                eval_data_list.append(json.loads(line))
        eval_data_list = self.sample_eval_data(eval_data_list, file_path)

        return eval_data_list

    def get_final_data_list(self, args, eval_data_list: List[Dict[str, Any]]):
        if args.broadcast:  # False
            group = self.broadcast_rank
            align_start_dp_rank = 0
        else:
            eval_data_list, group, align_start_dp_rank = get_final_list_dataset(
                eval_data_list,
                dist.get_world_size(),
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size
            )
        return eval_data_list, group, align_start_dp_rank

    def verify_data_format(self):
        for root_dir, sub_dirs, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith(".jsonl"):
                    raise ValueError(f"Not valid file. {file} not endwith .jsonl")
                file_path = os.path.join(root_dir, file)
                eval_data_list = self.read_jsonl_file(file_path)
                for data_item in eval_data_list:
                    CustomEval.validate_data_format(data_item)

    def top_k_eval(self):
        pass