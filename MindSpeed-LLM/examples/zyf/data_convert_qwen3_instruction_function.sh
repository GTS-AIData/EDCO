DATE=0807
TARGET_DIR="/data1/z30044758/Qwen3/dataset/${DATE}/output_function"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

python preprocess_data.py \
    --input  /data1/z30044758/Qwen3/dataset/${DATE}/function_v${DATE}.jsonl \
    --tokenizer-name-or-path /data1/z30044758/Qwen3/Model/Qwen3-4B/ \
    --output-prefix /data1/z30044758/Qwen3/dataset/${DATE}/output_function/generated_result_train \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 20 \
    --cache-dir /data1/zhouchang/Qwen3/dataset/tmp \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"messages":"data", "system": "meta_prompt", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant", "observation_tag":"tool", "function_tag":"assistant"}}'