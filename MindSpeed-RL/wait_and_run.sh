#!/bin/bash
# 文件名：wait_and_run.sh
CMD="bash ./examples/grpo/grpo_trainer_qwen3_4b.sh"
Time="12:15"
# 计算目标时间戳（今晚 20:15）
target=$(date -d $Time +%s)   # 若 macOS 用：target=$(date -j -f %H:%M 20:15 +%s)
now=$(date +%s)
# 如果已错过今晚 20:15，就改成明晚 20:15
if [ "$target" -le "$now" ]; then
   echo "已错过 $Time，脚本直接退出。"
   exit 0
fi

echo "等待到 $(date -d @$target '+%F %T') 再执行：$CMD"
sleep $((target - now))

# 开始干活
exec $CMD
