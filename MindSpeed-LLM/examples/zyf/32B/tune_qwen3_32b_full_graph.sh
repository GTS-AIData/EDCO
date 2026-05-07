#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=`echo $VC_WORKER_HOSTS|awk -F "," '{print $1}'`
master_addr=`echo $VC_WORKER_HOSTS|awk -F "," '{print $1}'`
echo ${MASTER_ADDR}
server_ip=`env|grep MA_CURRENT_HOST_IP |awk -F "=" '{print$2}'`
echo $server_ip #获取当前服务器的ip
ranknum=`grep -rns "$server_ip" /user/config/jobstart_hccl.json|awk -F "$server_ip" '{print$2}'|awk -F "rank_id" '{print$2}'|awk -F "\"" '{print$3}'`
echo $ranknum #获取当前服务器的rank序号，从0开始
server_index=`awk 'BEGIN{printf "%.0f\n",'${ranknum}'/8}'`
echo $sever_index #每台环境rank序号除以8 ，就是当前环境实际顺序
export WORLD_SIZE=32   #world_size设置 128
export MASTER_PORT=12345
export NNODES=4
export NPUS_PER_NODE=8
# please fill these path configurations
CKPT_LOAD_DIR="/home/ma-user/modelarts/user-job-dir/Qwen3/model_weight/Qwen3-32B-mcore-TP8PP4/"
CKPT_SAVE_DIR="/home/ma-user/modelarts/user-job-dir/Qwen3/train_result/$version/${TRAIN_TYPE}"
DATA_PATH="/home/ma-user/modelarts/user-job-dir/Qwen3/dataset/$version/output_${TRAIN_TYPE}/generated_result_train"
TOKENIZER_PATH="/home/ma-user/modelarts/user-job-dir/Qwen3/Model/Qwen3-32B/"

TP=8
PP=4
MBS=1
GBS=8
TRAIN_ITERS=
SAVE_ITERS=2000
SEQ_LENGTH=8192

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
    --num-layers 64 \
    --hidden-size 5120 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --num-attention-heads 64 \
    --ffn-hidden-size 25600 \
    --max-position-embeddings ${SEQ_LENGTH} \
    --seq-length ${SEQ_LENGTH} \
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
    --lr 1.25e-6 \
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
    --eval-interval ${TRAIN_ITERS} \
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

DISTRIBUTED_ARGS="--nproc_per_node $MA_NUM_GPUS --nnodes $MA_NUM_HOSTS --node_rank $VC_TASK_INDEX --master_addr $master_addr --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    $CKPT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_qwen3_32b_full_function.log