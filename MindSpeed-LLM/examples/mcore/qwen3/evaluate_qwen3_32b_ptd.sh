#!/bin/bash
# 请务必先阅读根目录下的llm_eval.md文档

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES="卡号设置"
# please fill these path configurations
TOKENIZER_PATH=""
CHECKPOINT=""
DATA_PATH="" # 推理数据输入路径，支持递归子文件
INFERENCE_OUT_PATH="" # 推理数据输出路径

TASK="custom_eval"
TASK_TYPE="general"
EVALUATION_BATCH_SIZE=3
MAX_TOKENS_TO_OOM=12000
MAX_NEW_TOKENS=128

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6007
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=4
SEQ_LENGTH=8192

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --task-data-path ${DATA_PATH} \
        --task ${TASK} \
        --task-type ${TASK_TYPE} \
        --output-dir ${INFERENCE_OUT_PATH} \
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-layers 64 \
        --hidden-size 5120 \
        --ffn-hidden-size 25600 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length ${SEQ_LENGTH} \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --max-new-tokens 128 \
        --max-position-embeddings 32768 \
        --disable-bias-linear \
        --swiglu \
        --norm-epsilon 1e-6 \
        --padded-vocab-size 151936 \
        --make-vocab-size-divisible-by 1 \
        --position-embedding-type rope \
        --load ${CHECKPOINT} \
        --hf-chat-template \
        --kv-channels 128 \
        --qk-layernorm \
        --untie-embeddings-and-output-weights \
        --norm-topk-prob \
        --rotary-base 1000000 \
        --use-rotary-position-embeddings \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --normalization RMSNorm \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-gradient-accumulation-fusion \
        --attention-softmax-in-fp32 \
        --tokenizer-not-use-fast \
        --exit-on-missing-checkpoint \
        --no-masked-softmax-fusion \
        --micro-batch-size 1 \
        --evaluation-batch-size ${EVALUATION_BATCH_SIZE} \
        --max-tokens-to-oom ${MAX_TOKENS_TO_OOM} \
        --no-load-rng \
        --no-load-optim \
        --seed 42 \
        --bf16 \
        | tee logs/evaluate_qwen3_32b_ptd.log
