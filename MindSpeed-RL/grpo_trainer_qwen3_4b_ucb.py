# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import math
import re
import subprocess

import numpy as np


class UCB1:
    def __init__(self, n_arms):
        """
        初始化UCB1算法

        参数:
        n_arms: 老虎机臂的数量
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的平均奖励
        self.total_counts = 0  # 总选择次数

    def sample(self):
        """
        根据UCB1公式选择臂

        返回:
        选择的臂的索引
        """
        # 确保每个臂至少被选择一次
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # 计算UCB值并选择最大的
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_counts) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
        更新选定臂的统计信息

        参数:
        chosen_arm: 被选择的臂的索引
        reward: 观察到的奖励值
        """
        self.total_counts += 1
        self.counts[chosen_arm] += 1

        # 更新平均值: new_avg = old_avg + (reward - old_avg) / count
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n


def extract_grpo_score_mean(log_file_path, total_iterations=20):
    grpo_means = []
    pattern = re.compile(r"iteration:\s*\d+\s*/\s*\d+.*?grpo/score/mean\s*:\s*([+-]?\d*\.?\d+)")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "iteration:" in line and "grpo/score/mean" in line:
                match = pattern.search(line)
                if match:
                    try:
                        value = float(match.group(1))
                        grpo_means.append(value)
                    except ValueError:
                        continue  # 跳过无法转换的行

    # 只取前 total_iterations 个（防止日志重复或多余）
    grpo_means = grpo_means[:total_iterations]

    if len(grpo_means) == 0:
        print("未找到任何 grpo/score/mean 记录。")
        return None

    average = sum(grpo_means) / len(grpo_means)
    print(f"共提取 {len(grpo_means)} 个 grpo/score/mean 值:")
    print(grpo_means)
    print(f"平均值: {average:.6f}")
    return average


def train_function(arm_index, term):
    print(f"在第{term}轮训练选择第{arm_index}条路径")
    home_dir = "/home"
    if term == 0:
        CKPT_LOAD_DIR = f"{home_dir}/Qwen3/model_weight/Qwen3-4B-mcore-TP8PP1/"
    else:
        CKPT_LOAD_DIR = f"{home_dir}/train_result/1117/dc_random_12000_ucb/models/round{term}_tuned_mc"

    CKPT_SAVE_DIR = f"{home_dir}/train_result/1117/dc_random_12000_ucb/models/round{int(term) + 1}_tuned_mc"
    output_prefix = f"{home_dir}/train_result/1117/dc_random_12000_ucb/processed_data/arm_{int(arm_index) + 1}"
    log_file_path = f"{home_dir}/train_result/1117/dc_random_12000_ucb/logs/grpo_qwen3_4b_round{term}.log"

    script_path = f"{home_dir}/Qwen3/MindSpeed-RL/examples/grpo/loop_ucb_grpo_trainer_qwen3_4b_main.sh"

    args = [
        CKPT_LOAD_DIR,
        CKPT_SAVE_DIR,
        output_prefix,
        str(term),
        log_file_path
    ]
    print(f"传入参数: {args}")
    # 调用 shell 脚本（串行执行，等待完成）
    result = subprocess.run(
        ["bash", script_path] + args,
        check=True,  # 如果脚本返回非0，抛出异常
        text=True,  # 使用文本模式（而非 bytes）
    )
    # 可选：检查返回码
    if result.returncode != 0:
        raise RuntimeError(f"Shell script failed with return code {result.returncode}")

    reward_score = extract_grpo_score_mean(log_file_path)
    return reward_score


if __name__ == '__main__':
    # 创建一个有3个臂的UCB1算法
    ucb = UCB1(n_arms=3)

    # === 手动注入前 19 轮（index 0~18）的状态 ===
    ucb.counts = np.array([6, 6, 7])
    ucb.values = np.array([0.538333, 0.556667, 0.577143])
    ucb.total_counts = 19

    print("已恢复至第 19 轮（即将执行 index=19）")
    print(f"当前 counts: {ucb.counts}")
    print(f"当前 values: {ucb.values}")

    # 从 index=19 继续执行到 24（共 25 轮）
    for index in range(19, 25):
        chosen_arm = ucb.sample()
        reward = train_function(chosen_arm, index)

        ucb.update(chosen_arm, reward)
        print(f"选择臂 {chosen_arm}, 获得奖励 {reward:.2f}")

    print("\n最终统计:")
    print(f"每个臂的选择次数: {ucb.counts}")
    print(f"每个臂的平均奖励: {ucb.values}")
