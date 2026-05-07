# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir "hf dir" \
    --save-dir "output dir" \
    --tokenizer-model "tokenizer path" \
    --params-dtype bf16 \
    --model-type-hf qwen3