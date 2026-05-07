#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <task_name>"
  exit 1
fi

export task_name="$1"
export MA_JOB_DIR=/home/Qwen3/
export INPUT_JSONL="/home/Qwen3/dataset/0919/${task_name}.jsonl"
export model_path="/home/Qwen3/Model/Qwen3-1.7B/"
export CKPT_LOAD_DIR="/home/Qwen3/model_weight/Qwen3-1.7B-mcore-TP8PP1"
export CKPT_SAVE_DIR="/home/train_result/0919/${task_name}"

# ========= 配置参数 =========
TOTAL_ROUNDS=5
samples_num=320
BASE_DATA_PATH=${INPUT_JSONL}       # 基础数据集
INITIAL_MC_MODEL_PATH="${CKPT_LOAD_DIR}"   # 初始 MCore 模型（用于训练）
INITIAL_HF_MODEL_PATH="${model_path}"       # 初始 HF 模型（用于第一轮数据筛选）

# 工作目录
LLM_DIR="${MA_JOB_DIR}/MindSpeed-LLM"
RL_DIR="${MA_JOB_DIR}/MindSpeed-RL"
PREPROCESS_CONFIG_DIR="${RL_DIR}/configs/datasets"
TRAIN_CONFIG_DIR="${RL_DIR}/configs"
DATA_OUTPUT_DIR="${CKPT_SAVE_DIR}/data"
MODEL_OUTPUT_ROOT="${CKPT_SAVE_DIR}/models"

# 日志文件
LOG_FILE="${CKPT_SAVE_DIR}/iterative_training.log"

# 创建必要目录
mkdir -p "$DATA_OUTPUT_DIR"
mkdir -p "$MODEL_OUTPUT_ROOT"

echo "$(date): 开始迭代训练流程，跳过第1轮，共执行 $((TOTAL_ROUNDS - 1)) 轮（第2~$TOTAL_ROUNDS轮）" | tee -a "$LOG_FILE"

# ========= 初始化模型路径：加载第1轮结果 =========
ROUND1_HF_DIR="${MODEL_OUTPUT_ROOT}/round1_tuned_hf"
ROUND1_MC_DIR="${MODEL_OUTPUT_ROOT}/round1_tuned_mc"

# 检查第一轮输出是否存在
if [ ! -d "$ROUND1_HF_DIR" ] || [ ! -d "$ROUND1_MC_DIR" ]; then
    echo "$(date): 错误：第一轮训练输出目录不存在！" | tee -a "$LOG_FILE"
    echo "$(date): 请确认以下路径存在：" | tee -a "$LOG_FILE"
    echo "HF: $ROUND1_HF_DIR" | tee -a "$LOG_FILE"
    echo "MC: $ROUND1_MC_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# 设置当前模型为第一轮训练结果
CURRENT_HF_MODEL_PATH="$ROUND1_HF_DIR"
CURRENT_MC_MODEL_PATH="$ROUND1_MC_DIR"

echo "$(date): 已成功加载第一轮训练结果：" | tee -a "$LOG_FILE"
echo "$(date):   数据筛选模型 (HF): $CURRENT_HF_MODEL_PATH" | tee -a "$LOG_FILE"
echo "$(date):   模型微调模型 (MC): $CURRENT_MC_MODEL_PATH" | tee -a "$LOG_FILE"

# ========= 外层大循环：从第2轮开始 =========
for round in $(seq 2 $TOTAL_ROUNDS); do
    echo "$(date): >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "$(date): 开始第 $round 轮训练"
    echo "$(date): 数据筛选模型 (HF): $CURRENT_HF_MODEL_PATH" | tee -a "$LOG_FILE"
    echo "$(date): 模型微调模型 (MC): $CURRENT_MC_MODEL_PATH" | tee -a "$LOG_FILE"

    # --- 路径定义 ---
    DATA_OUTPUT_JSONL="$DATA_OUTPUT_DIR/selected_data_round${round}.jsonl"
    ROUND_MC_DIR="${MODEL_OUTPUT_ROOT}/round${round}_tuned_mc"      # MC 保存路径
    ROUND_HF_DIR="${MODEL_OUTPUT_ROOT}/round${round}_tuned_hf"      # HF 转换后路径
    output_prefix="${DATA_OUTPUT_DIR}/round${round}"

    mkdir -p "$ROUND_MC_DIR"
    mkdir -p "$ROUND_HF_DIR"
    mkdir -p "$output_prefix"

    export DATA_OUTPUT_JSONL="$DATA_OUTPUT_JSONL"
    export ROUND_MC_DIR="$ROUND_MC_DIR"
    export ROUND_HF_DIR="$ROUND_HF_DIR"
    export output_prefix="$output_prefix"
    export CURRENT_MC_MODEL_PATH="$CURRENT_MC_MODEL_PATH"

    # ========= 步骤1: 基于熵的数据筛选（使用 HF 模型） =========
    cd "$LLM_DIR" || { echo "$(date): 错误：无法进入工作目录: $LLM_DIR"; exit 1; }
    echo "$(date): [Round $round] 步骤1: 执行基于熵的数据筛选..." | tee -a "$LOG_FILE"
    STEP_START=$(date +%s)

    python data_selection_by_entropy.py \
        --model_path "$CURRENT_HF_MODEL_PATH" \
        --data_path "$BASE_DATA_PATH" \
        --output_path "$DATA_OUTPUT_JSONL" \
        --samples_num $samples_num

    STEP_END=$(date +%s)
    STEP_TIME=$((STEP_END - STEP_START))
    if [ ! -f "$DATA_OUTPUT_JSONL" ] || [ ! -s "$DATA_OUTPUT_JSONL" ]; then
        echo "$(date): 错误：数据筛选未生成有效文件 $DATA_OUTPUT_JSONL" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "$(date): [Round $round] 步骤1 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))，输出: $DATA_OUTPUT_JSONL" | tee -a "$LOG_FILE"

    # ========= 步骤2: 数据处理 =========
    echo "$(date): [Round $round] 步骤2: 执行数据处理脚本..." | tee -a "$LOG_FILE"
    STEP_START=$(date +%s)

    cd "$RL_DIR" || { echo "$(date): 错误：无法进入工作目录: $RL_DIR"; exit 1; }
    PREPROCESS_CONFIG_NAME=deepscaler_${task_name}_round${round}
    PREPROCESS_CONFIG_YAML="${PREPROCESS_CONFIG_DIR}/${PREPROCESS_CONFIG_NAME}.yaml"

    python ./cli/generate_temp_config.py \
      configs/datasets/deepscaler_loop.yaml \
      "$PREPROCESS_CONFIG_YAML"

    if [ $? -ne 0 ]; then
      echo "❌ Failed to generate preprocess config" | tee -a "$LOG_FILE"
      exit 1
    fi
    echo "✅ Preprocess config saved: $PREPROCESS_CONFIG_YAML"

    echo "🚀 Running data preprocessing..." | tee -a "$LOG_FILE"
    bash ./examples/data/preprocess_data.sh "${PREPROCESS_CONFIG_NAME}"

    STEP_END=$(date +%s)
    STEP_TIME=$((STEP_END - STEP_START))
    if [ $? -ne 0 ]; then
        echo "$(date): 错误：数据处理脚本执行失败" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "$(date): [Round $round] 步骤2 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))" | tee -a "$LOG_FILE"

    # ========== 步骤3: 模型RL（使用 MCore 模型） ==========
    echo "$(date): [Round $round] 步骤3: 开始模型微调..." | tee -a "$LOG_FILE"
    STEP_START=$(date +%s)

    TRAIN_CONFIG_NAME=grpo_qwen3_1.7B_A3_${task_name}_round${round}
    TRAIN_CONFIG_YAML="${TRAIN_CONFIG_DIR}/${TRAIN_CONFIG_NAME}.yaml"

    python ./cli/generate_temp_config.py \
      configs/grpo_qwen3_1.7b_A3_loop.yaml \
      "$TRAIN_CONFIG_YAML"

    if [ $? -ne 0 ]; then
      echo "❌ Failed to generate training config" | tee -a "$LOG_FILE"
      exit 1
    fi
    echo "✅ Training config saved: $TRAIN_CONFIG_YAML" | tee -a "$LOG_FILE"

    echo "🚀 Running GRPO training..." | tee -a "$LOG_FILE"
    bash ./examples/grpo/grpo_trainer_qwen3_1.7b_loop.sh "${TRAIN_CONFIG_NAME}"

    STEP_END=$(date +%s)
    STEP_TIME=$((STEP_END - STEP_START))
    if [ $? -ne 0 ]; then
        echo "$(date): 错误：模型RL失败" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "$(date): [Round $round] 步骤3 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))，保存至: $ROUND_MC_DIR" | tee -a "$LOG_FILE"

    # ========= 步骤4: MCore → HF 格式转换（用于下一轮数据筛选） =========
    echo "$(date): [Round $round] 步骤4: 转换模型格式为 Hugging Face..." | tee -a "$LOG_FILE"
    STEP_START=$(date +%s)

    cd "$LLM_DIR" || { echo "$(date): 错误：无法进入工作目录: $LLM_DIR"; exit 1; }
    conda run -n mindspeed_llm_v1 bash "$LLM_DIR/examples/zyf/1.7B/ckpt_convert_qwen3_mcore2hf_loop.sh" \
        "$ROUND_MC_DIR" \
        "$ROUND_HF_DIR"

    STEP_END=$(date +%s)
    STEP_TIME=$((STEP_END - STEP_START))
    if [ $? -ne 0 ]; then
        echo "$(date): 错误：模型格式转换失败" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "$(date): [Round $round] 步骤4 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))，输出: $ROUND_HF_DIR" | tee -a "$LOG_FILE"

    # ========= 更新模型路径 =========
    CURRENT_HF_MODEL_PATH="$ROUND_HF_DIR"
    CURRENT_MC_MODEL_PATH="$ROUND_MC_DIR"

    echo "$(date): 第 $round 轮完成，更新模型路径:"
    echo "  下一轮数据筛选模型: $CURRENT_HF_MODEL_PATH"
    echo "  下一轮微调模型: $CURRENT_MC_MODEL_PATH"
done

echo "$(date): 所有迭代训练完成！"
echo "最终 HF 模型: $CURRENT_HF_MODEL_PATH"
echo "最终 MC 模型: $CURRENT_MC_MODEL_PATH"
