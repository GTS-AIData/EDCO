# json_output nl2sql graph function
DATE=0210
FOLDER_NAME=graph
HOME_DIR="/home"
TARGET_DIR="${HOME_DIR}/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

python preprocess_data.py \
    --input  ${HOME_DIR}/Qwen3/dataset/${DATE}/${FOLDER_NAME}_v${DATE}.jsonl \
    --tokenizer-name-or-path ${HOME_DIR}/Qwen3/Model/Qwen3-32B/ \
    --output-prefix ${HOME_DIR}/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}/generated_result_train \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 20 \
    --cache-dir ${HOME_DIR}/Qwen3/dataset/tmp \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"messages":"data", "system": "meta_prompt", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant", "observation_tag":"tool"}}'
