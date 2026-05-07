# json_output nl2sql graph
DATE=0807
FOLDER_NAME=graph
TARGET_DIR="/data1/zhouchang/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

python preprocess_data.py \
    --input  /data1/zhouchang/Qwen3/dataset/${DATE}/${FOLDER_NAME}_v${DATE}.jsonl \
    --tokenizer-name-or-path /data1/zhouchang/Qwen3/Model/Qwen3-4B/ \
    --output-prefix /data1/zhouchang/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}/generated_result_train \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 20 \
    --cache-dir /data1/zhouchang/Qwen3/dataset/tmp \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"prompt":"instruction","query":"input","response":"output"}'