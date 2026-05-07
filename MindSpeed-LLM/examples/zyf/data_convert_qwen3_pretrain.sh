DATE=0905
FOLDER_NAME=dc_base
TARGET_DIR="/data1/zhouchang/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# --input /data1/zhouchang/Qwen3/dataset/${DATE}/${FOLDER_NAME}_v${DATE}.jsonl
python preprocess_data.py \
 --input /data1/zhouchang/Qwen3/dataset/temp.jsonl \
 --tokenizer-name-or-path /data1/zhouchang/Qwen3/Model/Qwen3-8B/ \
 --output-prefix /data1/zhouchang/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}/processed \
 --handler-name GeneralPretrainHandler \
 --tokenizer-type PretrainedFromHF \
 --workers 4 \
 --n-subs 8 \
 --cache-dir /data5/zhouchang/cache/ \
 --json-keys text \
 --log-interval 1000
