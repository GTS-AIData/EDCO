#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_IF_BASE_PORT=25809
export NPU_ASD_ENABLE=0
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=7000
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6032
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 参数检查
if [ $# -ne 4 ]; then
    echo "用法: $0 <CKPT_LOAD_DIR> <CKPT_SAVE_DIR> <DATA_PATH_PREFIX> <samples_num>"
    echo "例如: $0 /path/to/mc_model /path/to/save_dir /path/to/data/processed 300"
    echo "注意: DATA_PATH_PREFIX 应指向 preprocess 生成的 'xxx_text_document' 文件的前缀部分"
    exit 1
fi

# please fill these path configurations
CKPT_LOAD_DIR=$1
CKPT_SAVE_DIR=$2
DATA_PATH=$3
TOKENIZER_PATH="${model_path}"
samples_num=$4

TP=8
PP=1
MBS=2
GBS=8
TRAIN_ITERS=$(( samples_num  / GBS ))
echo "TRAIN_ITERS: $TRAIN_ITERS"
SAVE_ITERS=20000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 36 \
    --hidden-size 2560 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --num-attention-heads 32 \
    --ffn-hidden-size 9728 \
    --max-position-embeddings 32768 \
    --seq-length 4096 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --disable-bias-linear \
    --train-iters ${TRAIN_ITERS} \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --min-lr 1.25e-7 \
    --lr 2.5e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_ITERS} \
    --eval-interval ${SAVE_ITERS} \
    --eval-iters 0 \
"

CKPT_ARGS="
    --no-save-optim \
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type qwen \
    --variable-seq-lengths
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    $CKPT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee ${CKPT_SAVE_DIR}/tune_qwen3_4b_full_loop.log