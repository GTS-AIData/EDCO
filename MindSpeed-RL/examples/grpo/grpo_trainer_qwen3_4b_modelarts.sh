#!/bin/bash
# 环境变量设置
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

export HCCL_IF_BASE_PORT=64000
export TASK_QUEUE_ENABLE=2
export HCCL_IF_BASE_PORT=24703

#修改为对应主节点IP
HOSTS="$VC_TASK_HOSTS"
MASTER_HOST="${HOSTS%%,*}"

# worker0作为主节点启动ray集群，其他worker加入到集群
if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** work-0 starts"
  ray start --head --port 6344 --dashboard-host=0.0.0.0 -node-ip-address=MASTER_HOST --dashboard-port=8260 --resources='{"NPU": 8}'
  sleep 60
else
  echo "********** work-$VC_TASK_INDEX starts"
  echo "$MASTER_HOST:6344"
  sleep 30
  ray start --address="$MASTER_HOST:6344" --resources='{"NPU": 8}'
  sleep 30
  # 非0结点循环检查ray集群状态
  while true; do
    ray status > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      break
    fi
    sleep 30
  done
fi
ray status

# worker0 启动训练任务
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_PATH="${CKPT_LOAD_DIR}/qwen3_4b_rl_${TRAIN_TYPE}.log"
echo ${LOG_PATH}
if [ "$VC_TASK_INDEX" = "0" ]; then
  echo "********** work-0 training"
  sleep 1m
  ray status
  python cli/train_grpo.py --config-name=grpo_qwen3_4_A3 2>&1 | tee ${LOG_PATH}
  # 结束ray集群
  ray stop
fi

ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;

ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}
