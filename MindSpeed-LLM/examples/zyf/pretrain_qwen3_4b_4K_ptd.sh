DATE=0828
FOLDER_NAME=pre_train
#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,7

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
# CKPT_LOAD_DIR="/data1/zhouchang/Qwen3/model_weight/Qwen3-4B-mcore-TP4PP1/"
CKPT_LOAD_DIR="/data1/zhouchang/Qwen3/train_result/${DATE}/${FOLDER_NAME}"
CKPT_SAVE_DIR="/data1/zhouchang/Qwen3/train_result/${DATE}/${FOLDER_NAME}"
DATA_PATH="/data1/zhouchang/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}/generated_result_train_text_document"
TOKENIZER_PATH="/data1/zhouchang/Qwen3/Model/Qwen3-4B/"

# 设置是否为首次训练（true = 首次，false = 续训）
# FIRST_TRAIN=true
FIRST_TRAIN=false   # 断点续训时打开这行，注释上一行

TP=4
PP=1
CP=1
MBS=1
GBS=1
SEQ_LENGTH=4096
TRAIN_ITERS=300
SAVE_ITERS=50
CP_TYPE='ulysses_cp_algo'
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --qk-layernorm \
    --kv-channels 128 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 36 \
    --hidden-size 2560 \
    --ffn-hidden-size 9728 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0
"

# 只有首次训练时才添加 no-load 参数
if [ "$FIRST_TRAIN" = "true" ]; then
    OUTPUT_ARGS="$OUTPUT_ARGS \
        --no-load-optim \
        --no-load-rng"
fi

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    | tee logs/train_mcore_qwen3_4b.log
