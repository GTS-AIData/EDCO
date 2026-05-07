export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ $# -ne 2 ]; then
    echo "用法: $0 <mc_model_input_dir> <hf_model_output_dir>"
    exit 1
fi

MC_DIR=$1
HF_DIR=$2

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ${MC_DIR} \
    --save-dir ${model_path} \
    --params-dtype bf16 \
    --model-type-hf llama2 \

if [ $? -ne 0 ]; then
    echo "转换失败"
    exit 1
fi

echo "MCore → HF 转换完成: $HF_DIR"

echo "移动模型至：$HF_DIR"
mv ${model_path}/mg2hf/* ${HF_DIR}/
# cp ${model_required_file}/* ${HF_DIR}/
cp ${model_path}/model_required_file/* ${HF_DIR}/
