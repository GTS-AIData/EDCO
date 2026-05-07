#!/bin/bash

set -euo pipefail

# ============= 配置区 =============
DATE=1021
HOME_DIR=/home
TRAIN_ROOT="${HOME_DIR}/train_result/${DATE}"
CONVERT_SCRIPT="./examples/zyf/batch_ckpt_convert_qwen3_mcore2hf.sh"
TARGET_BASE="${HOME_DIR}/Qwen3/Model/${DATE}"

# 检查
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "错误：转换脚本不存在: $CONVERT_SCRIPT"
    exit 1
fi

if [ ! -d "$TRAIN_ROOT" ]; then
    echo "错误：训练结果目录不存在: $TRAIN_ROOT"
    exit 1
fi


echo "开始批量转换，根目录: $TRAIN_ROOT"

# ============= 遍历所有子目录 =============
for FOLDER_PATH in "$TRAIN_ROOT"/*/; do
    # 跳过非目录
    [ ! -d "$FOLDER_PATH" ] && continue

    FOLDER_NAME=$(basename "$FOLDER_PATH")

    # ========== 新增：跳过指定名称的目录 ==========
    case "$FOLDER_NAME" in
     dc_random_12000_common|\
     dc_random_12000_loss|\
     wl_random_12000_common|\
     wl_random_12000_sorted_by_perplexity|\
     wl_random_12000_loss|\
     wl_random_12000_sorted_by_instruction_length)
            echo "⏭️  跳过目录: $FOLDER_NAME"
            continue
            ;;
    esac
    # ==========================================

    echo "=================================================="
    echo "正在处理: $FOLDER_NAME"
    echo "时间戳: $(date '+%Y-%m-%d %H:%M:%S')"

    # ✅ 正确方式：将 FOLDER_NAME 作为参数传入
    if bash "$CONVERT_SCRIPT" "$FOLDER_NAME" "$DATE"; then
        echo "✅ 成功转换: $FOLDER_NAME"
    else
        echo "❌ 转换失败: $FOLDER_NAME"
    fi
done

echo "=================================================="
echo "✅ 所有模型转换任务完成！结果保存在: $TARGET_BASE"