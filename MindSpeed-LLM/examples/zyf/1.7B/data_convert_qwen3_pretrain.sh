cd /home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}
mkdir output_${TRAIN_TYPE}

cd ${MA_JOB_DIR}/Qwen3/MindSpeed-LLM
python preprocess_data.py \
    --input  /home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}/${TRAIN_TYPE}_v${version}.jsonl \
    --tokenizer-name-or-path /home/ma-user/modelarts/user-job-dir/Qwen3/Model/Qwen3-1.7B/ \
    --output-prefix /home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}/output_${TRAIN_TYPE}/${TRAIN_TYPE}_text_document \
    --handler-name GeneralPretrainHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 40 \
    --json-keys text \
    --log-interval 1000