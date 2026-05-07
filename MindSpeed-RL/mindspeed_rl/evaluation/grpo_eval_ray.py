# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# mindspeed_rl/utils/ray_eval_helper.py
import os
import random
import ray
import subprocess


@ray.remote(resources={"NPU": 8}, max_retries=0)   # 失败就拉倒，别重试
def ray_evaluate_one_ckpt(ckpt_path: str, global_step: int):
    """在 Ray 集群里占 8 卡跑单机评测，完全独立通信域"""
    env = os.environ.copy()
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    master_port = 16000 + random.randint(0, 999)          # 端口隔离
    out_dir = f"/home/train_result/1010/grpo_eval/step_{global_step}"
    log_file = f"/home/train_result/1010/grpo_eval/step_{global_step}/eval.log"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "torchrun",
        "--nproc_per_node", "8",
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", "127.0.0.1",
        "--master_port", str(master_port),
        "evaluation.py",
        "--spec", "mindspeed_llm.tasks.models.spec.qwen3_spec", "layer_spec",
        "--task_data_path", "/home/Qwen3/dataset/1010/dc_eval",
        "--task", "custom_eval",
        "--task_type", "general",
        "--output_dir", out_dir,
        "--use_mcore_models",
        "--tensor_model_parallel_size", "8",
        "--pipeline_model_parallel_size", "1",
        "--num_layers", "36",
        "--hidden_size", "2560",
        "--ffn_hidden_size", "9728",
        "--num_attention_heads", "32",
        "--group_query_attention",
        "--num_query_groups", "8",
        "--seq_length", "8192",
        "--max_new_tokens", "512",
        "--max_position_embeddings", "32768",
        "--disable_bias_linear",
        "--swiglu",
        "--norm_epsilon", "1e-6",
        "--padded_vocab_size", "151936",
        "--make_vocab_size_divisible_by", "1",
        "--position_embedding_type", "rope",
        "--load", ckpt_path,           # ← 关键：传进来的新 ckpt
        "--hf_chat_template",
        "--kv_channels", "128",
        "--qk_layernorm",
        "--norm_topk_prob",
        "--rotary_base", "1000000",
        "--use_rotary_position_embeddings",
        "--tokenizer_type", "PretrainedFromHF",
        "--tokenizer_name_or_path", "/home/Qwen3/Model/Qwen3_4B/",
        "--normalization", "RMSNorm",
        "--attention_dropout", "0.0",
        "--hidden_dropout", "0.0",
        "--no_gradient_accumulation_fusion",
        "--attention_softmax_in_fp32",
        "--tokenizer_not_use_fast",
        "--exit_on_missing_checkpoint",
        "--no_masked_softmax_fusion",
        "--micro_batch_size", "1",
        "--evaluation_batch_size", "8",
        "--max_tokens_to_oom", "65536",
        "--no_load_rng",
        "--no_load_optim",
        "--seed", "42",
        "--bf16",
    ]

    completed = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,  # 捕获输出
        stderr=subprocess.STDOUT  # 合并 stderr
    )

    # tee 实时写文件 & 打印
    subprocess.run(
        ["tee", log_file],
        input=completed.stdout,
        stderr=subprocess.STDOUT
    )
    return log_file

