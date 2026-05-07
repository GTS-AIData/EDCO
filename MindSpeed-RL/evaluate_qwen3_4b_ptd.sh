#!/bin/bash
# 请务必先阅读根目录下的llm_eval.md文档

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# please fill these path configurations
TOKENIZER_PATH="/home/Qwen3/Model/Qwen3_4B/"
CHECKPOINT="/home/Qwen3/model_weight/Qwen3_4B_mcore_TP8PP1"
DATA_PATH="/home/Qwen3/dataset/1010/dc_eval"
INFERENCE_OUT_PATH="/home/train_result/1010/dc_eval/"

TASK="custom_eval"
TASK_TYPE="general"
EVALUATION_BATCH_SIZE=8
MAX_TOKENS_TO_OOM=32384
MAX_NEW_TOKENS=512

# Change for multinode config
MASTER_ADDR="localhost"
MASTER_PORT=6008
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
TP=8
PP=1
SEQ_LENGTH=8192

DISTRIBUTED_ARGS="
    __nproc_per_node $NPUS_PER_NODE \
    __nnodes $NNODES \
    __node_rank $NODE_RANK \
    __master_addr $MASTER_ADDR \
    __master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
        __spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        __task_data_path ${DATA_PATH} \
        __task ${TASK} \
        __task_type ${TASK_TYPE} \
        __output_dir ${INFERENCE_OUT_PATH} \
        __use_mcore_models \
        __tensor_model_parallel_size ${TP} \
        __pipeline_model_parallel_size ${PP} \
        __num_layers 36 \
        __hidden_size 2560 \
        __ffn_hidden_size 9728 \
        __num_attention_heads 32 \
        __group_query_attention \
        __num_query_groups 8 \
        __seq_length ${SEQ_LENGTH} \
        __max_new_tokens ${MAX_NEW_TOKENS} \
        __max_position_embeddings 32768 \
        __disable_bias_linear \
        __swiglu \
        __norm_epsilon 1e_6 \
        __padded_vocab_size 151936 \
        __make_vocab_size_divisible_by 1 \
        __position_embedding_type rope \
        __load ${CHECKPOINT} \
        __hf_chat_template \
        __kv_channels 128 \
        __qk_layernorm \
        __norm_topk_prob \
        __rotary_base 1000000 \
        __use_rotary_position_embeddings \
        __tokenizer_type PretrainedFromHF \
        __tokenizer_name_or_path ${TOKENIZER_PATH} \
        __normalization RMSNorm \
        __attention_dropout 0.0 \
        __hidden_dropout 0.0 \
        __no_gradient_accumulation_fusion \
        __attention_softmax_in_fp32 \
        __tokenizer_not_use_fast \
        __exit_on_missing_checkpoint \
        __no_masked_softmax_fusion \
        __micro_batch_size 1 \
        __evaluation_batch_size ${EVALUATION_BATCH_SIZE} \
        __max_tokens_to_oom ${MAX_TOKENS_TO_OOM} \
        __no_load_rng \
        __no_load_optim \
        __seed 42 \
        __bf16 \
        | tee logs/evaluate_qwen3_4b_ptd.log
