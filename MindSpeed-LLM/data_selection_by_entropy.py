#!/usr/bin/env python3
# 文件名: data_selection_by_entropy.py
# 功能: 基于熵的数据筛选（支持文件/目录）

import random
import argparse
import os
import json
import numpy as np
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import ray
from tqdm import tqdm
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

# 设置环境变量
os.environ["VLLM_USE_V1"] = "1"


def merge_jsonl_files(input_dir, output_file):
    """将目录下所有 .jsonl 文件合并为一个文件"""
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在或不是目录: {input_dir}")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"目录中未找到任何 .jsonl 文件: {input_dir}")

    print(f"发现 {len(jsonl_files)} 个 .jsonl 文件，正在合并...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in jsonl_files:
            print(f"  处理: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + '\n')

    print(f"合并完成: {output_file}")


def get_input_path(data_path_str):
    """
    解析输入路径：
    - 如果是文件 → 返回该文件路径
    - 如果是目录 → 合并后返回合并文件路径
    """
    path = Path(data_path_str).resolve()

    if path.is_file() and path.suffix == '.jsonl':
        print(f"[数据加载] 输入为单个文件: {path}")
        return path

    elif path.is_dir():
        # 自动生成合并文件名：{dirname}.merged.jsonl
        merged_filename = f"{path.name}.merged.jsonl"
        # 使用临时目录存放合并文件，避免污染
        merged_file = Path(tempfile.gettempdir()) / merged_filename

        print(f"[数据加载] 输入为目录: {path}")
        print(f"[数据加载] 正在合并为临时文件: {merged_file}")

        merge_jsonl_files(path, merged_file)

        return merged_file

    else:
        raise ValueError(f"无效的输入路径（不是 .jsonl 文件或目录）: {data_path_str}")


def add_output_to_selected(selected_data, original_lines):
    """
    为选中的数据添加 output 字段

    参数:
    selected_data: 同事函数返回的数据列表（包含完整前缀字符串）
    original_lines: 原始数据行列表

    返回:
    添加了 output 字段的数据列表
    """
    prefix = "请你用最多不超过三句话，回答下面的问题，如果需要必要的推理，请你尽可能减少推理过程。\n问题：\n"

    # 创建 instruction 到 output 的映射
    instruction_to_output = {}
    for line in original_lines:
        data = json.loads(line)
        instruction_to_output[data["instruction"]] = data.get("output", "")

    # 为选中的数据添加 output 字段
    result_with_output = []
    missing_output_count = 0

    for item in selected_data:
        # 从完整字符串中提取原始 instruction
        full_text = item["instruction"]
        if full_text.startswith(prefix):
            instruction = full_text[len(prefix):]

            if instruction in instruction_to_output:
                item_with_output = item.copy()
                # 使用原始 instruction（不带前缀）
                item_with_output["instruction"] = instruction
                item_with_output["output"] = instruction_to_output[instruction]
                # 添加空的 input 字段
                item_with_output["input"] = ""
                result_with_output.append(item_with_output)
            else:
                print(f"警告: 找不到 instruction '{instruction}' 对应的 output")
                missing_output_count += 1
        else:
            print(f"警告: 字符串 '{full_text}' 不以预期前缀开头")
            missing_output_count += 1

    if missing_output_count > 0:
        print(f"警告: 总共 {missing_output_count} 条数据没有找到匹配的 output")

    return result_with_output


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSONL文件中读取数据。
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的softmax实现。"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def calculate_entropy_for_response(response: Any) -> Dict[str, Any]:
    """
    计算单个响应的平均熵。
    """
    response_entropies = []
    if not response.outputs:
        return None

    for token_lp_dict in response.outputs[0].logprobs:
        if token_lp_dict:
            log_probs = np.array([lp.logprob for lp in token_lp_dict.values()])
            probs = softmax(log_probs)
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            response_entropies.append(entropy)

    if response_entropies:
        response_entropies_arr = np.array(response_entropies)
        k = max(1, int(0.2 * len(response_entropies_arr)))
        sorted_indices = np.argsort(response_entropies_arr)[-k:]
        top_entropy = response_entropies_arr[sorted_indices]
        entropy_mean = np.mean(top_entropy)
        return {
            "instruction": response.prompt,
            "entropy_mean": entropy_mean
        }
    return None


# 使用 @ray.remote 装饰器，并指定自定义资源 "NPU"
@ray.remote(resources={"NPU": 1})
def worker_process(
        device_id: int,
        model_path: str,
        prompts: List[str],
        micro_batch_size: int
):
    """
    Ray 远程任务。在 Ray 节点上加载模型并执行推理。
    """
    print(f"Ray 进程 {os.getpid()} 正在启动，使用 NPU {device_id}...")
    # 在 Ray 进程中设置设备环境变量
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
    try:
        llm = LLM(
            model=model_path,
            device="npu",
            enforce_eager=True
        )
        print(f"Ray 进程 {os.getpid()} 模型加载成功。")

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.7,
            max_tokens=128,
            logprobs=20,
        )

        all_entropies = []
        total_prompts = len(prompts)

        # 核心改动：使用微批次处理
        for i in range(0, total_prompts, micro_batch_size):
            micro_batch_prompts = prompts[i:i + micro_batch_size]

            # 执行推理
            outputs = llm.generate(micro_batch_prompts, sampling_params)

            # 使用线程池进行后处理（计算熵）
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(calculate_entropy_for_response, out) for out in outputs]
                for future in futures:
                    result = future.result()
                    if result:
                        all_entropies.append(result)

        return all_entropies

    except Exception as e:
        print(f"Ray 进程 {os.getpid()} 出现错误: {e}")
        return []


def run_parallel_inference_with_ray(instructions, model_path, top_k_count, num_devices=8, micro_batch_size=40):
    # 1. 初始化 Ray 并设置自定义资源
    ray.init(resources={"NPU": num_devices})
    print("Ray 正在运行...")

    # 2. 读取数据集并按卡数切分
    total_prompts = len(instructions)
    chunk_size = total_prompts // num_devices
    chunks = [instructions[i:i + chunk_size] for i in range(0, total_prompts, chunk_size)]

    # 3. 提交任务到 Ray 节点
    futures = [worker_process.remote(i, model_path, chunks[i], micro_batch_size) for i in range(num_devices)]

    # 4. 在主进程中显示进度条并等待所有任务结束
    all_entropies = []
    with tqdm(total=total_prompts, desc="推理进度") as pbar:
        while futures:
            # 等待第一个完成的任务
            done_futures, futures = ray.wait(futures)
            # 收集结果
            results = ray.get(done_futures[0])
            all_entropies.extend(results)
            # 更新进度条
            pbar.n = len(all_entropies)
            pbar.refresh()

    # 5. 对所有结果进行排序
    if not all_entropies:
        print("未获取到任何熵值结果。")
        return

    sorted_entropies = sorted(all_entropies, key=lambda x: x["entropy_mean"], reverse=True)

    return sorted_entropies


def main():
    parser = argparse.ArgumentParser(description="基于熵的数据筛选（支持文件/目录）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="输入数据集路径 (.jsonl) 或包含多个 .jsonl 的目录")
    parser.add_argument("--output_path", type=str, required=True, help="输出数据集路径 (.jsonl)")
    parser.add_argument("--samples_num", type=int, default=1000, help="抽取样本数量 (默认: 1000)")
    # parser.add_argument("--topk_count", type=int, default=200, help="抽取样本数量 (默认: 1000)")

    args = parser.parse_args()

    try:
        # 统一获取最终要处理的 .jsonl 文件
        final_input_file = get_input_path(args.data_path)

        # 读取合并后的文件
        with open(final_input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            raise ValueError("输入数据为空")

        num_to_sample = min(args.samples_num, len(lines))
        print(f"准备从 {len(lines)} 条数据中抽取 {num_to_sample} 条样本")

        # 构建前缀
        prefix = "请你用最多不超过三句话，回答下面的问题，如果需要必要的推理，请你尽可能减少推理过程。\n问题：\n"

        # 创建带有前缀的指令列表
        instructions = []
        for i, line in enumerate(lines):
            data = json.loads(line)
            if "instruction" not in data:
                print(f"警告: 第 {i + 1} 行数据缺少 instruction 字段")
            else:
                instructions.append(prefix + data["instruction"])

        # 调用函数计算熵并选择高熵样本
        sorted_entropies = run_parallel_inference_with_ray(instructions, args.model_path, num_to_sample)
        print(f"返回 {len(sorted_entropies)} 条数据")

        # 为选中的数据添加 output 字段
        final_data = add_output_to_selected(sorted_entropies, lines)
        print(f"成功为 {len(final_data)} 条数据添加了 output 字段")

        # strip_samples_num = int(len(lines) * 0.05)
        # print(f"过滤掉前5%的数据，约{strip_samples_num}条")
        strip_samples_num = 0
        sorted_entropies_top_k = final_data[strip_samples_num: strip_samples_num + num_to_sample]
        print(f"成功从 {len(final_data)} 条数据中抽取{num_to_sample} 条数据")

        # 生成输出文件名（包含熵和不包含熵的两个版本）
        output_path = Path(args.output_path)
        output_with_entropy = output_path.parent / (output_path.stem + "_with_entropy" + output_path.suffix)

        # 写入包含熵的文件
        with open(output_with_entropy, 'w', encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 创建不包含熵的数据
        final_data_without_entropy = []
        for item in sorted_entropies_top_k:
            item_without_entropy = {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"]
            }
            final_data_without_entropy.append(item_without_entropy)

        # 写入不包含熵的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in final_data_without_entropy:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"包含熵的结果已保存到: {output_with_entropy}")
        print(f"不包含熵的结果已保存到: {output_path}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
