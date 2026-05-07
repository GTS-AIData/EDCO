cd /home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}
if [ ! -d "output_${TRAIN_TYPE}" ]; then
    mkdir -p "output_${TRAIN_TYPE}"
fi

cd ${MA_JOB_DIR}/Qwen3/MindSpeed-LLM
python preprocess_data.py \
    --input  /home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}/${TRAIN_TYPE}_v${version}.jsonl \
    --tokenizer-name-or-path /home/ma-user/modelarts/user-job-dir/Qwen3/Model/Qwen3-32B/ \
    --output-prefix /home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}/output_${TRAIN_TYPE}/generated_result_train \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 20 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"messages":"data", "system": "meta_prompt", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant", "observation_tag":"tool", "function_tag":"assistant"}}}'