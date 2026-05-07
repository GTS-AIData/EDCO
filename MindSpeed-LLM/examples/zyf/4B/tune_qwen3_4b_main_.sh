#!/bin/bash
export task_name=$1
export MA_JOB_DIR=/home/Qwen3/
export INPUT_JSONL=/home/Qwen3/dataset/0325/${task_name}.jsonl
export model_path=/home/Qwen3/Model/Qwen3-4B/
export CKPT_LOAD_DIR=/home/Qwen3/model_weight/Qwen3-4B-mcore-TP8PP1/
export CKPT_SAVE_DIR=/home/train_result/0325/${task_name}

# ========= 配置参数 =========
epoch=2
data_number=7000
samples_num=$(( data_number * epoch))
BASE_DATA_PATH=${INPUT_JSONL}       # 基础数据集
INITIAL_MC_MODEL_PATH="${CKPT_LOAD_DIR}"   # 初始 MCore 模型（用于训练）

# 工作目录
WORK_DIR="${MA_JOB_DIR}/MindSpeed-LLM"
DATA_OUTPUT_DIR="${CKPT_SAVE_DIR}/data"
MODEL_OUTPUT_ROOT="${CKPT_SAVE_DIR}/models"

# 日志文件
LOG_FILE="${CKPT_SAVE_DIR}/iterative_training.log"

# 创建必要目录
mkdir -p "$DATA_OUTPUT_DIR"
mkdir -p "$MODEL_OUTPUT_ROOT"


#  ========= 初始化模型路径 =========
# 当前用于微调的模型（MCore 格式）
CURRENT_MC_MODEL_PATH="$INITIAL_MC_MODEL_PATH"

# ========= 外层大循环 =========
echo "$(date): 模型微调模型 (MC): $CURRENT_MC_MODEL_PATH" | tee -a "$LOG_FILE"

# --- 路径定义 ---
DATA_OUTPUT_JSONL="$DATA_OUTPUT_DIR/selected_data_round${round}.jsonl"
ROUND_MC_DIR="${MODEL_OUTPUT_ROOT}/round${round}_tuned_mc"      # MC 保存路径
ROUND_HF_DIR="${MODEL_OUTPUT_ROOT}/round${round}_tuned_hf"      # HF 转换后路径
PROCESSED_DATA_PREFIX="${DATA_OUTPUT_DIR}/round${round}"

mkdir -p "$ROUND_MC_DIR"
mkdir -p "$ROUND_HF_DIR"
mkdir -p "$PROCESSED_DATA_PREFIX"


# ========= 步骤2: 数据处理 =========
echo "$(date): [Round $round] 步骤2: 执行数据处理脚本..." | tee -a "$LOG_FILE"
STEP_START=$(date +%s)

#bash "${MA_JOB_DIR}/MindSpeed-LLM/examples/zyf/4B/data_convert_instruction_sft_loop.sh" \
#    "$DATA_OUTPUT_JSONL" \
#    "${PROCESSED_DATA_PREFIX}/processed"

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
if [ $? -ne 0 ]; then
    echo "$(date): 错误：数据处理脚本执行失败" | tee -a "$LOG_FILE"
    exit 1
fi
echo "$(date): [Round $round] 步骤2 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))" | tee -a "$LOG_FILE"

# ========= 步骤3: 模型微调（使用 MCore 模型） =========
echo "$(date): [Round $round] 步骤3: 开始模型微调..." | tee -a "$LOG_FILE"
STEP_START=$(date +%s)

# echo "命令如下："
# echo "bash ${MA_JOB_DIR}/MindSpeed-LLM/examples/zyf/4B/tune_qwen3_4b_full_loop.sh $CURRENT_MC_MODEL_PATH $ROUND_MC_DIR $PROCESSED_DATA_PREFIX/processed_text_document"
#bash "${MA_JOB_DIR}/MindSpeed-LLM/examples/zyf/4B/tune_qwen3_4b_full_loop.sh" \
#    "$CURRENT_MC_MODEL_PATH" \
#    "$ROUND_MC_DIR" \
#    "$PROCESSED_DATA_PREFIX/processed"\
#    "$samples_num"

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
if [ $? -ne 0 ]; then
    echo "$(date): 错误：模型微调失败" | tee -a "$LOG_FILE"
    exit 1
fi
echo "$(date): [Round $round] 步骤3 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))，保存至: $ROUND_MC_DIR" | tee -a "$LOG_FILE"

# ========= 步骤4: MCore → HF 格式转换（用于下一轮数据筛选） =========
echo "$(date): [Round $round] 步骤4: 转换模型格式为 Hugging Face..." | tee -a "$LOG_FILE"
STEP_START=$(date +%s)

conda run -n mindspeed_llm_v1 bash "${MA_JOB_DIR}/MindSpeed-LLM/examples/zyf/4B/ckpt_convert_qwen3_mcore2hf.sh" \
    "$ROUND_MC_DIR" \
    "$ROUND_HF_DIR"

STEP_END=$(date +%s)
STEP_TIME=$((STEP_END - STEP_START))
if [ $? -ne 0 ]; then
    echo "$(date): 错误：模型格式转换失败" | tee -a "$LOG_FILE"
    exit 1
fi
echo "$(date): [Round $round] 步骤4 完成，耗时: $(printf '%dm %ds' $((STEP_TIME/60)) $((STEP_TIME%60)))，输出: $ROUND_HF_DIR" | tee -a "$LOG_FILE"
