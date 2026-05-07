#!/bin/bash

# 要执行的脚本
SCRIPT="./examples/zyf/4B/loop_tune_qwen3_4b_main.sh"

# 定义配对关系：(finetune_dataset, pretrain_ckpt)
# wl 开头的用 pre_train_wl_100，dc 开头的用 pre_train_dc_100
pairs=(
    "wl_random_12000 pre_train_wl_100"
    "wl_top_score_12000 pre_train_wl_100"
    "dc_random_12000 pre_train_dc_100"
    "dc_top_score_12000 pre_train_dc_100"
)

# 循环执行每一对
for pair in "${pairs[@]}"; do
    dataset_name=$(echo $pair | awk '{print $1}')
    pretrain_name=$(echo $pair | awk '{print $2}')
    echo "Running: $SCRIPT $dataset_name $pretrain_name"
    bash "$SCRIPT" "$dataset_name" "$pretrain_name"
done

echo "All tasks completed."