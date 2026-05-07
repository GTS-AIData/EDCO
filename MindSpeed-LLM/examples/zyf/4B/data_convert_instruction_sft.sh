cd ${MA_JOB_DIR}
mkdir output
cd ./output
mkdir ${TRAIN_TYPE}

cd ${MA_JOB_DIR}/MindSpeed-LLM || exit 1
python preprocess_data.py \
    --input "${INPUT_JSONL}" \
    --tokenizer-name-or-path "${model_path}" \
    --output-prefix "${MA_JOB_DIR}/output/${TRAIN_TYPE}/processed" \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 1 \
    --n-subs 1 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"prompt":"instruction","query":"input","response":"output"}'
