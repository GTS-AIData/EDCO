#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <task_name>"
  exit 1
fi

export task_name="$1"
export INPUT_JSONL="/home/Qwen3/dataset/0919/${task_name}.jsonl"
export model_path="/home/Qwen3/Model/Qwen3-1.7B/"
export CKPT_LOAD_DIR="/home/Qwen3/model_weight/Qwen3-1.7B-mcore-TP8PP1"
export CKPT_SAVE_DIR="/home/train_result/0919/${task_name}"
export output_prefix="/home/train_result/0919/${task_name}/processed_data"

# 创建输出目录
mkdir -p "$CKPT_SAVE_DIR"
mkdir -p "$output_prefix"

echo "✅ Environment variables set:"
echo "task_name: $task_name"
echo "INPUT_JSONL: $INPUT_JSONL"
echo "model_path: $model_path"
echo "CKPT_LOAD_DIR: $CKPT_LOAD_DIR"
echo "CKPT_SAVE_DIR: $CKPT_SAVE_DIR"
echo "output_prefix: $output_prefix"

# 检查 input 文件是否存在
if [ ! -f "$INPUT_JSONL" ]; then
  echo "❌ Input file not found: $INPUT_JSONL"
  exit 1
fi

PREPROCESS_CONFIG_DIR="/home/Qwen3/MindSpeed-RL/configs/datasets"
PREPROCESS_CONFIG_NAME=deepscaler_${task_name}
TRAIN_CONFIG_DIR="/home/Qwen3/MindSpeed-RL/configs"
TRAIN_CONFIG_NAME=grpo_qwen3_1.7B_A3_${task_name}
# ✅ 将配置文件保存在任务目录下，长期保留
PREPROCESS_CONFIG_YAML="${PREPROCESS_CONFIG_DIR}/${PREPROCESS_CONFIG_NAME}.yaml"
TRAIN_CONFIG_YAML="${TRAIN_CONFIG_DIR}/${TRAIN_CONFIG_NAME}.yaml"

echo "📁 Configuration files will be saved to: $CKPT_SAVE_DIR"
echo "   - Preprocess: $(basename "$PREPROCESS_CONFIG_YAML")"
echo "   - Training:   $(basename "$TRAIN_CONFIG_YAML")"

# ========== 第一步：生成预处理配置 ==========
python ./cli/generate_temp_config.py \
  configs/datasets/deepscaler_loop.yaml \
  "$PREPROCESS_CONFIG_YAML"

if [ $? -ne 0 ]; then
  echo "❌ Failed to generate preprocess config"
  exit 1
fi
echo "✅ Preprocess config saved: $PREPROCESS_CONFIG_YAML"

echo "🚀 Running data preprocessing..."
# python ./cli/preprocess_data.py --config_path="${CKPT_SAVE_DIR}" --config_name="${PREPROCESS_CONFIG_YAML}"
bash ./examples/data/preprocess_data.sh "${PREPROCESS_CONFIG_NAME}"

if [ $? -ne 0 ]; then
  echo "❌ Data preprocessing failed."
  exit 1
fi

# ========== 第二步：生成训练配置 ==========
python ./cli/generate_temp_config.py \
  configs/grpo_qwen3_1.7b_A3_loop.yaml \
  "$TRAIN_CONFIG_YAML"

if [ $? -ne 0 ]; then
  echo "❌ Failed to generate training config"
  exit 1
fi
echo "✅ Training config saved: $TRAIN_CONFIG_YAML"

echo "🚀 Running GRPO training..."
bash ./examples/grpo/grpo_trainer_qwen3_1.7b_loop.sh "${TRAIN_CONFIG_NAME}"

if [ $? -ne 0 ]; then
  echo "❌ Training script failed."
  exit 1
fi

echo "🎉 Task '$task_name' completed successfully!"