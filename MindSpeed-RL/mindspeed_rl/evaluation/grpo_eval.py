# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import time
import ray
from distutils import dist
from torch import distributed as dist

from evaluation import custom_eval, LLMChat
# from megatron.training.arguments import parse_args
# from megatron.training.global_vars import set_global_variables
# from mindspeed_llm.tasks.evaluation.utils import add_text_generate_args
from mindspeed_llm.tasks.inference import MegatronModuleForCausalLM

from mindspeed_rl.utils import Loggers

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


# @ray.remote
# class EvalActor:
#     def __init__(self, actor_config, eval_model_provider, tokenizer):
#         # 1. 构造 Megatron 全局 args（只在该 Actor 进程执行一次）
#         ignore_unknown_args = {
#             'no_load_rng': True,
#             'no_load_optim': True
#         }
#         megatron_args = parse_args(add_text_generate_args, ignore_unknown_args)
#         cfg_dict = actor_config if isinstance(actor_config, dict) else actor_config.dict()
#
#         # 2. 字段级覆盖
#         for k, v in cfg_dict.items():
#             if hasattr(megatron_args, k):
#                 setattr(megatron_args, k, v)
#
#         # 3. 补充推理专用字段
#         megatron_args.data_parallel_size = 1
#         megatron_args.task = "custom_eval"
#         megatron_args.task_type = "general"
#         megatron_args.eval_batch_size = 2
#         megatron_args.max_tokens_to_oom = 32384
#         megatron_args.max_new_tokens = 512
#         megatron_args.hf_chat_template = True
#
#         # 4. 注册到全局
#         set_global_variables(megatron_args)
#
#         self.megatron_args = megatron_args
#         self.eval_model_provider = eval_model_provider
#         self.tokenizer = tokenizer
#
#     def eval_latest(self, logger):
#         """Ray 远程入口，每次保存后调用"""
#         # 3. 真正推理
#         grpo_eval(
#             self.megatron_args,
#             self.eval_model_provider,
#             self.tokenizer,
#             logger=logger
#         )
#         return


def grpo_eval(megatron_args, model, tokenizer, logger):
    megatron_args = parse_args(add_text_generate_args, True)
    logger.info(f"megatron_args: {megatron_args}")

    # model = MegatronModuleForCausalLM.from_pretrained(
    #     model_provider=eval_model_provider,
    #     pretrained_model_name_or_path=megatron_args.save
    # )

    rank = dist.get_rank()
    a = time.time()
    custom_eval(megatron_args, LLMChat(megatron_args, model, tokenizer))
    if rank == 0:
        logger.info(f"Custom_eval Running Time: {time.time() - a}")


if __name__ == '__main__':
    # TOKENIZER_PATH = "/home/Qwen3/Model/Qwen3-4B/"
    # CHECKPOINT = "/home/Qwen3/model_weight/Qwen3-4B-mcore-TP8PP1"
    # DATA_PATH = "/home/Qwen3/dataset/1010_eval"
    # INFERENCE_OUT_PATH = "/home/train_result/1010/dc_eval/"
    test_logger = Loggers('grpo_trainer_hybrid_eval')
    grpo_eval(actor_config, eval_model_provider, tokenizer, test_logger)
