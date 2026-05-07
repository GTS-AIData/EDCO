cd ${MA_JOB_DIR}
mkdir ${output_prefix}

cd ${MA_JOB_DIR}/MindSpeed-LLM
python preprocess_data.py \
    --input ${input_json} \
    --tokenizer-name-or-path ${model_path} \
    --output-prefix ${output_prefix}/processed \
    --handler-name GeneralPretrainHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 32 \
    --n-subs 64 \
    --cache-dir ${MA_JOB_DIR}/cache \
    --json-keys text \
    --log-interval 1000