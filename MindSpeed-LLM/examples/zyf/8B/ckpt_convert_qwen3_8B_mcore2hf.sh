# json_output nl2sql function graph

DATE=0209
FOLDER_NAME=function

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
    --save-dir /opt/zhouchang/Qwen3/Model/Qwen3-8B/ \
    --params-dtype bf16 \
    --model-type-hf qwen3

TARGET_DIR="/opt/zhouchang/Qwen3/Model/${DATE}"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

mkdir /opt/zhouchang/Qwen3/Model/${DATE}/${FOLDER_NAME}
mv /opt/zhouchang/Qwen3/Model/Qwen3-8B/mg2hf/* /opt/zhouchang/Qwen3/Model/${DATE}/${FOLDER_NAME}/
cp /opt/zhouchang/Qwen3/Model/模型挪动文件/* /opt/zhouchang/Qwen3/Model/${DATE}/${FOLDER_NAME}/