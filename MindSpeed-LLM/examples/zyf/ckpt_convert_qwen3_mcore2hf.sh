# json_output nl2sql function graph
# pre_train_100 pre_train_75 pre_train_50 pre_train_33 pre_train_20 pre_train_16
# doc_baike_case doc_baike doc_case baike_case
DATE=0911
FOLDER_NAME=graph

source /opt/tangxian/packages/cann/rc1_b120/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /opt/zhouchang/train_result/${DATE}/${FOLDER_NAME} \
    --save-dir /opt/zhouchang/Qwen3/Model/Qwen3-4B/ \
    --params-dtype bf16 \
    --model-type-hf qwen3

TARGET_DIR="/opt/zhouchang/Qwen3/Model/${DATE}"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

mkdir ${TARGET_DIR}/${FOLDER_NAME}
mv /opt/zhouchang/Qwen3/Model/Qwen3-4B/mg2hf/* /opt/zhouchang/Qwen3/Model/${DATE}/${FOLDER_NAME}/
cp /opt/zhouchang/Qwen3/Model/模型挪动文件/* /opt/zhouchang/Qwen3/Model/${DATE}/${FOLDER_NAME}/
