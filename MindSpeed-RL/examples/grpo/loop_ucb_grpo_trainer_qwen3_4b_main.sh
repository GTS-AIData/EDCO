#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <CKPT_LOAD_DIR>"
  exit 1
fi

export MASTER_ADDR="10.50.89.138"
export model_path="/home/Qwen3/Model/Qwen3-4B/"
export CKPT_LOAD_DIR=$1
export CKPT_SAVE_DIR=$2
export output_prefix=$3
export term=$4
export log_file_path=$5
export task_name="dc_ucb"

# 创建输出目录
mkdir -p "$CKPT_SAVE_DIR"

TRAIN_CONFIG_DIR="/home/Qwen3/MindSpeed-RL/configs"
TRAIN_CONFIG_NAME=grpo_qwen3_4B_A3_${task_name}_${term}
# ✅ 将配置文件保存在任务目录下，长期保留
TRAIN_CONFIG_YAML="${TRAIN_CONFIG_DIR}/${TRAIN_CONFIG_NAME}.yaml"

echo "📁 Configuration files will be saved to: $CKPT_SAVE_DIR"
echo "   - Training:   $(basename "$TRAIN_CONFIG_YAML")"

# ========== 第二步：生成训练配置 ==========
python ./cli/generate_temp_config.py \
  configs/grpo_qwen3_4b_A3_ucb_loop.yaml \
  "$TRAIN_CONFIG_YAML"

if [ $? -ne 0 ]; then
  echo "❌ Failed to generate training config"
  exit 1
fi
echo "✅ Training config saved: $TRAIN_CONFIG_YAML"

echo "🚀 Running GRPO training..."
bash ./examples/grpo/grpo_trainer_qwen3_4b_loop_ucb.sh "${TRAIN_CONFIG_NAME}" "${MASTER_ADDR}" "${log_file_path}"

if [ $? -ne 0 ]; then
  echo "❌ Training script failed."
  exit 1
fi

echo "🎉 Task '$task_name' completed successfully!"
