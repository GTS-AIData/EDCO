DATE=0919
FOLDER_NAME=wl_top_score_12000_sorted_by_instruction_length

export CUDA_DEVICE_MAX_CONNECTIONS=1

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /home/train_result/${DATE}/${FOLDER_NAME} \
    --save-dir /home/Qwen3/Model/Qwen3-1.7B/ \
    --params-dtype bf16 \
    --model-type-hf qwen3

TARGET_DIR="/home/Qwen3/Model/${DATE}"
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

mkdir /home/Qwen3/Model/${DATE}/${FOLDER_NAME}
mv /home/Qwen3/Model/Qwen3-1.7B/mg2hf/* /home/Qwen3/Model/${DATE}/${FOLDER_NAME}/
cp /home/Qwen3/Model/模型挪动文件/* /home/Qwen3/Model/${DATE}/${FOLDER_NAME}/

