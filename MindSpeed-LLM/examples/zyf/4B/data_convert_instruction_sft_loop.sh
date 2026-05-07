#!/bin/bash

if [ $# -ne 2 ]; then
    echo "用法: $0 <input_jsonl> <output_prefix_dir>"
    echo "例如: $0 /path/to/data.jsonl /path/to/output_dir"
    exit 1
fi

INPUT_JSONL=$1
OUTPUT_PREFIX=$2

cd ${MA_JOB_DIR}/MindSpeed-LLM || exit 1

python preprocess_data.py \
    --input "${INPUT_JSONL}" \
    --tokenizer-name-or-path "${model_path}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 8 \
    --n-subs 16 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"prompt":"instruction","query":"input","response":"output"}'

if [ $? -ne 0 ]; then
    echo "数据处理失败"
    exit 1
fi
