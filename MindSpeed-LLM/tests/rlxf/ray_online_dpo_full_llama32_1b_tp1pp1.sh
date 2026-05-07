#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True


basepath=$(cd `dirname $0`; cd ../../../; pwd)


python $basepath/ray_gpt.py --config-dir=$basepath/tests/pipeline/configs --config-name=ray_online_dpo_full_llama32_1b_tp1pp1