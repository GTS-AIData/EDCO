cd ${MA_JOB_DIR}
mkdir /models/data/pretrain_data/z00928231/${TRAIN_TYPE}/

cd ${MA_JOB_DIR}/MindSpeed-LLM
python preprocess_data.py \
    --input ${input_jsonl} \
    --tokenizer-name-or-path ${model_path} \
    --output-prefix /models/data/pretrain_data/z00928231/${TRAIN_TYPE}/processed \
    --handler-name GeneralPretrainHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 32 \
    --n-subs 64 \
    --cache-dir ${MA_JOB_DIR}/cache \
    --json-keys text \
    --log-interval 1000
