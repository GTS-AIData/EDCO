DATE=0807
FOLDER_NAME=graph
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,7

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=7500
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/data1/zhouchang/Qwen3/model_weight/Qwen3-4B-mcore-TP4PP1/"
CKPT_SAVE_DIR="/data1/zhouchang/Qwen3/train_result/${DATE}/${FOLDER_NAME}"
DATA_PATH="/data1/zhouchang/Qwen3/dataset/${DATE}/output_${FOLDER_NAME}/generated_result_train"
TOKENIZER_PATH="/data1/zhouchang/Qwen3/Model/Qwen3-4B/"

# 设置是否为首次训练（true = 首次，false = 续训）
FIRST_TRAIN=true
# FIRST_TRAIN=false   # 断点续训时打开这行，注释上一行

TP=4
PP=1
MBS=1
GBS=8
SEQ_LENGTH=4096
TRAIN_ITERS=100
SAVE_ITERS=20

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
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --min-lr 1.25e-7 \
    --lr 1.25e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
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
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
"

TUNE_ARGS="
    --stage sft \
    --is-instruction-dataset \
    --prompt-type qwen \
    --variable-seq-lengths
"

# 只有首次训练时才添加 no-load 参数
if [ "$FIRST_TRAIN" = "true" ]; then
    OUTPUT_ARGS="$OUTPUT_ARGS \
        --no-load-optim \
        --no-load-rng"
    TUNE_ARGS="$TUNE_ARGS \
        --finetune"
fi

CKPT_ARGS="
    --no-save-optim \
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
    | tee logs/tune_qwen3_4b_full_haichuan.log
