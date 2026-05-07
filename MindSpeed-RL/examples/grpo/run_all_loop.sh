#!/bin/bash

# 要执行的脚本
SCRIPT="./examples/grpo/loop_grpo_trainer_qwen3_4b_main.sh"
MASTER_ADDR="10.50.89.132"
pretrain_ckpt=pre_train_wl_100
# 定义数据集列表
datasets=(
    "wl_random_12000_common"
    "wl_random_12000_sorted_by_instruction_length"
    "wl_random_12000_answer_complexity"
    "wl_random_12000_sorted_by_perplexity"
)

# 循环执行每一个数据集
for dataset_name in "${datasets[@]}"; do
    echo "Running: $SCRIPT $dataset_name $pretrain_ckpt"
    bash "$SCRIPT" "$dataset_name" "$MASTER_ADDR" "$pretrain_ckpt"
done

echo "All tasks completed."