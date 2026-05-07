#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model load ckpt path"

TP=1
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 30 \
    --hidden-size 2560 \
    --load ${CKPT_LOAD_DIR} \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --embed-layernorm \
    --padded-vocab-size 250880 \
    --make-vocab-size-divisible-by 1 \
    --attention-softmax-in-fp32 \
    --apply-query-key-layer-scaling \
    --lr 1.2e-6 \
    --train-iters 1000 \
    --init-method-std 0.0048 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --position-embedding-type alibi \
    --normalization LayerNorm \
    --min-lr 1e-8 \
    --lr-decay-iters 430000 \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42 
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 1 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --jit-compile \
    --save $CKPT_SAVE_DIR \
    | tee logs/train_bloom_3b.log
