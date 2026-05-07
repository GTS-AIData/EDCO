#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=`echo $VC_WORKER_HOSTS|awk -F "," '{print $1}'`
master_addr=`echo $VC_WORKER_HOSTS|awk -F "," '{print $1}'`
echo ${MASTER_ADDR}
server_ip=`env|grep MA_CURRENT_HOST_IP |awk -F "=" '{print$2}'`
echo $server_ip #获取当前服务器的ip
ranknum=`grep -rns "$server_ip" /user/config/jobstart_hccl.json|awk -F "$server_ip" '{print$2}'|awk -F "rank_id" '{print$2}'|awk -F "\"" '{print$3}'`
echo $ranknum #获取当前服务器的rank序号，从0开始
server_index=`awk 'BEGIN{printf "%.0f\n",'${ranknum}'/8}'`
echo $sever_index #每台环境rank序号除以8 ，就是当前环境实际顺序
export WORLD_SIZE=8   #world_size设置 128
export MASTER_PORT=12345

# please fill these path configurations
CKPT_LOAD_DIR="/home/ma-user/modelarts/user-job-dir/Qwen3/model_weight/Qwen3-1.7B-mcore-TP8PP1/"
CKPT_SAVE_DIR="/home/ma-user/modelarts/user-job-dir/Qwen3/train_result/$version/${TRAIN_TYPE}/"
DATA_PATH="/home/ma-user/modelarts/user-job-dir/Qwen3/dataset/${version}/output_${TRAIN_TYPE}/${TRAIN_TYPE}_text_document"
TOKENIZER_PATH="/home/ma-user/modelarts/user-job-dir/Qwen3/Model/Qwen3-1.7B/"

TP=8
PP=1
MBS=1
GBS=128
SEQ_LENGTH=8192
TRAIN_ITERS=34721
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $MA_NUM_GPUS \
    --nnodes $MA_NUM_HOSTS \
    --node_rank $VC_TASK_INDEX \
    --master_addr $master_addr \
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
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --qk-layernorm \
    --kv-channels 128 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 28 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 16 \
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
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

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
    | tee logs/pretrain_qwen3_1point7b_${TRAIN_TYPE}.log
