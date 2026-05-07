# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir /home/Qwen3/Model/llama3.2-3B/ \
   --save-dir /home/Qwen3/model_weight/llama3.2-3B-mcore-TP8PP1/ \
   --tokenizer-model /home/Qwen3/Model/llama3.2-3B/tokenizer.json \
   --model-type-hf llama2 \
   --params-dtype bf16