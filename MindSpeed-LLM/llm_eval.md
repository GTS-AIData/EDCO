# 1.背景
实现训练后推理，且支持自定义数据集。

# 2.支持模型
原则上，支持MindSpeed-LLM框架训练的模型，均可实现训练后推理功能，仅需针对具体的模型适配chat模板即可，当前已经针对Qwen3系列适配。
单batch推理时，权重切分支持DP、PP以及TP
多batch推理时，权重切分支持DP、TP

# 3.使用介绍
首先部署MindSpeed相关环境，参考部署文档：docs/pytorch/install_guide.md
其次，以qwen3系列为例进行介绍：
## 3.1 权重切分
权重切分建议使用TP

## 3.2 自定义数据集格式
参考AI数据部OPB领域数据交付的标准数据格式，wiki如下：https://wiki.huawei.com/domains/27396/wiki/47203/WIKI202507117463797
基于多轮对话，已经适配qwen3源码的chat模板，确保推理精度

## 3.3 启动脚本
bash examples/mcore/qwen3/evaluate_qwen3_4b_ptd.sh

# 3.4 脚本参数设置
- TASK="custom_eval"
自定义推理任务（建议保持不变）。与框架的gsm8k、mmlu等开源数据集评估区分开来，
- TASK_TYPE="general"
支持任务类型。 general、function_call、jsonoutput、nl2sql、knowledge_graph_extraction
备注：原则上非上述也支持，但务必保证格式与上述一致
- EVALUATION_BATCH_SIZE=1
推理的batch_size大小
- MAX_TOKENS_TO_OOM
显式的内存安全检查，变量充当阈值。（实际评估时，需结合卡的剩余值观察调整）
- MAX_NEW_TOKENS
最大输出token数量

# 4. 后续todo
1.多机推理待测试。
无物理机，等待中
