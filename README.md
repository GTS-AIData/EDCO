---

# 🚀 EDCO-Mindie 强化学习训练框架

本项目基于 **Mindie** 框架实现强化学习（RL）训练。核心依赖于昇腾（Ascend）提供的 **MindSpeed-LLM** 与 **MindSpeed-RL** 加速库。

[![Ascend](https://img.shields.io/badge/Ascend-MindSpeed-orange)](https://github.com/Ascend)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 📌 项目依赖

在开始之前，请确保已参考昇腾官方仓库获取最新信息：
* **MindSpeed-LLM**: [GitHub 仓库](https://github.com/Ascend/MindSpeed-LLM)
* **MindSpeed-RL**: [GitHub 仓库](https://github.com/Ascend/MindSpeed-RL)

---

## 🛠️ 环境搭建

### 1. 容器部署
本项目依赖昇腾官方提供的 Docker 镜像。请先从官网下载 `MindSpeed-RL` 对应的 Docker 容器包。

**导入并启动镜像：**
```bash
# 导入镜像
docker import mindspeed_rl_v1.tar mindspeed_rl_zhouchang

# 启动容器
docker run -id -u root --name mindspeed_rl_zc \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  --privileged=true \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
  -v /usr/local/sbin/:/usr/local/sbin \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -v /opt/DPC/zhouchang/:/home \
  --shm-size=1000g \
  --net=host \
  -w /home/ \
  mindspeed_rl_zhouchang \
  /bin/bash 

# 进入容器
docker exec -it mindspeed_rl_zc bash
```

### 2. 配置虚拟环境
进入容器后，需要配置专门的 Conda 环境并安装依赖：

```bash
# 创建并激活环境
conda create -n mindspeed_llm_v1 --clone mindspeed_rl_v1
conda activate mindspeed_llm_v1

# 安装 MindSpeed 加速库
cd /root/packages/MindSpeed
pip install -r requirements.txt 
pip install -e .

# 安装 MindSpeed-LLM 依赖
cd /home/Qwen3/MindSpeed-LLM
pip install -r requirements.txt
```

---

## 🏋️ 训练启动指南

### 1. 启动脚本
以 **Qwen3-4B** 为例，主启动脚本位于 MindSpeed-RL 目录下。执行时需传入任务名称与机器 IP。

```bash
# 进入 RL 目录
cd /home/Qwen3/MindSpeed-RL

# 执行启动脚本
# 用法: bash <脚本路径> <任务名称> <机器IP>
bash ./examples/grpo/loop_course_grpo_trainer_qwen3_4b_main.sh my_task_01 7.246.80.0
```

> [!TIP]
> 若使用其他模型，请对应修改 `MindSpeed-RL` 提供的配置 YAML 文件。

### 2. 核心配置说明
脚本内部主要参数配置如下，请根据实际物理路径进行调整：

| 变量名 | 描述 | 备注 |
| :--- | :--- | :--- |
| `task_name` | 任务名称 | 脚本第一个参数 |
| `MASTER_ADDR` | 机器 IP | 脚本第二个参数 |
| `INPUT_JSONL` | 基础训练数据集 | 需确保路径存在 |
| `model_path` | HF 格式模型路径 | 用于第一轮数据筛选 |
| `CKPT_LOAD_DIR` | MCore 格式模型路径 | 用于 RL 训练加载 |
| `TOTAL_ROUNDS` | 训练总轮数 | 默认: 5 |
| `samples_num` | 每轮采样数量 | 默认: 200 |

---

## ⚖️ 裁判模型与校验规则修改

项目支持 **规则校验** 或 **模型校验**。你可以根据业务需求自定义裁判逻辑。

* **配置文件路径**: `mindspeed_rl/models/rule_verifier.py`
* **当前配置**: 默认开启模型校验。
* **修改建议**: 
    * 搜索函数 `ask_qwen35_grade_score` 进行逻辑调整。
    * 确保校验函数的输出格式符合 GRPO 的奖励反馈要求。

---

> [!IMPORTANT]
> **路径检查**: 在启动前，请务必检查 `INITIAL_MC_MODEL_PATH` 和 `INITIAL_HF_MODEL_PATH` 对应的权重文件是否完整。
