# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /opt/tangxian/packages/cann/rc1_b120/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /opt/zhouchang/Qwen3/Model/Qwen3-8B/ \
    --save-dir /opt/zhouchang/Qwen3/model_weight/Qwen3-8B-mcore-TP8PP1/ \
    --tokenizer-model ./model_from_hf/qwen3_hf/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3
