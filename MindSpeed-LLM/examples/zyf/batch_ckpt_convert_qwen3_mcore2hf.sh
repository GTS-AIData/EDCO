#!/bin/bash
set -euo pipefail

# ============= 读取参数 =============
if [ $# -ne 2 ]; then
    echo "用法: $0 <folder_name> <date>"
    echo "示例: $0 dc_doc_case 0912"
    exit 1
fi

FOLDER_NAME="$1"
DATE="$2"

echo ">>> 开始转换模型: $FOLDER_NAME (日期: $DATE)"
HOME_DIR=/home
# 其余脚本内容保持不变（使用 $FOLDER_NAME 和 $DATE）
#source /opt/tangxian/packages/cann/rc1_b120/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir "${HOME_DIR}/train_result/${DATE}/${FOLDER_NAME}" \
    --save-dir "${HOME_DIR}/Qwen3/Model/Qwen3-4B/" \
    --params-dtype bf16 \
    --model-type-hf qwen3

# 创建目标目录并移动文件
TARGET_DIR="${HOME_DIR}/Qwen3/Model/${DATE}"
mkdir -p "$TARGET_DIR"
OUTPUT_DIR="$TARGET_DIR/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

mv "${HOME_DIR}/Qwen3/Model/Qwen3-4B/mg2hf/"* "$OUTPUT_DIR/"
cp "${HOME_DIR}/Qwen3/Model/模型挪动文件/"* "$OUTPUT_DIR/"
