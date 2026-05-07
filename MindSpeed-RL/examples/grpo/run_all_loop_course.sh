#!/bin/bash

# 要执行的脚本
SCRIPT="./examples/grpo/loop_course_grpo_trainer_qwen3_1.7b_main.sh"

# 定义配对关系：(finetune_dataset, pretrain_ckpt)
# wl 开头的用 pre_train_wl_100，dc 开头的用 pre_train_dc_100
pairs=(
#    "wl_top_score_12000"
#    "wl_top_score_12000_common"
#    "wl_top_score_12000_sorted_by_instruction_length"
#    "wl_top_score_12000_answer_complexity"
#    "wl_top_score_12000_sorted_by_perplexity"
    "dc_random_12000"
    "dc_top_score_12000"
#    "dc_top_score_12000_common"
#    "dc_top_score_12000_sorted_by_instruction_length"
#    "dc_top_score_12000_answer_complexity"
#    "dc_top_score_12000_sorted_by_perplexity"

)

# 循环执行每一对
for pair in "${pairs[@]}"; do
    dataset_name=$(echo $pair | awk '{print $1}')
    echo "Running: $SCRIPT $dataset_name"
    bash "$SCRIPT" "$dataset_name"
done

echo "All tasks completed."