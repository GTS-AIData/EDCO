# EDCO

Official implementation of **EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning**, accepted at ICML 2026.

This repository provides an Ascend/MindSpeed implementation of EDCO for domain-specific LLM fine-tuning. The current release focuses on Qwen3-based GRPO training and implements a multi-round curriculum loop that selects informative domain samples, fine-tunes the model, converts checkpoints, and uses the updated model to orchestrate the next round of data selection.

## Overview

EDCO is designed for iterative domain adaptation of large language models. Each training round performs:

1. Entropy-based sample selection from the domain training pool.
2. Dataset preprocessing for MindSpeed-RL.
3. GRPO fine-tuning with rule/model-based reward verification.
4. MCore-to-Hugging Face checkpoint conversion.
5. Curriculum update for the next round using the newly tuned model.

The implementation is built on:

- [MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM) for LLM training utilities, checkpoint conversion, and entropy-based data selection.
- [MindSpeed-RL](https://github.com/Ascend/MindSpeed-RL) for GRPO training on Ascend NPUs.
- Ray for distributed rollout and training orchestration.

## Repository Layout

```text
.
|-- MindSpeed-LLM/
|   |-- data_selection_by_entropy.py
|   `-- examples/zyf/
|-- MindSpeed-RL/
|   |-- configs/
|   |-- examples/grpo/
|   `-- mindspeed_rl/models/rule_verifier.py
|-- LICENSE
`-- README.md
```

Key files:

- `MindSpeed-LLM/data_selection_by_entropy.py`: entropy-based curriculum sample selection.
- `MindSpeed-RL/examples/grpo/loop_course_grpo_trainer_qwen3_4b_main.sh`: end-to-end EDCO training loop for Qwen3-4B.
- `MindSpeed-RL/configs/grpo_qwen3_4b_A3_course_loop.yaml`: GRPO training template.
- `MindSpeed-RL/configs/datasets/deepscaler_course_loop.yaml`: preprocessing template for selected samples.
- `MindSpeed-RL/mindspeed_rl/models/rule_verifier.py`: reward verification logic.

## Requirements

The code is intended for Ascend NPU environments and follows the dependency stack used by MindSpeed-LLM and MindSpeed-RL. Before running EDCO, prepare a working MindSpeed container or host environment with:

- Ascend driver, firmware, Toolkit, Kernel, and NNAL versions compatible with the MindSpeed release being used.
- Python and PyTorch/torch-npu versions compatible with the Ascend software stack.
- MindSpeed, MindSpeed-LLM, and MindSpeed-RL installed in editable or source form.
- Ray, vLLM with Ascend support, and the Python dependencies listed in the MindSpeed subprojects.

Refer to the upstream installation guides for the most current compatibility matrix:

- MindSpeed-LLM: https://github.com/Ascend/MindSpeed-LLM
- MindSpeed-RL: https://github.com/Ascend/MindSpeed-RL

## Environment Setup

The following commands show the expected setup pattern inside an Ascend-enabled MindSpeed container. Adjust paths and environment names for your cluster.

```bash
conda create -n mindspeed_llm_v1 --clone mindspeed_rl_v1
conda activate mindspeed_llm_v1

cd /root/packages/MindSpeed
pip install -r requirements.txt
pip install -e .

cd /home/Qwen3/MindSpeed-LLM
pip install -r requirements.txt

cd /home/Qwen3/MindSpeed-RL
pip install -r requirements.txt
```

For container deployment, mount the Ascend driver, firmware, shared memory, model directory, dataset directory, and output directory required by your cluster. The EDCO scripts assume that `MindSpeed-LLM` and `MindSpeed-RL` are available under the same workspace root.

## Running EDCO

The main Qwen3-4B training loop is:

```bash
cd /home/Qwen3/MindSpeed-RL
bash ./examples/grpo/loop_course_grpo_trainer_qwen3_4b_main.sh <task_name> <master_ip>
```

Example:

```bash
bash ./examples/grpo/loop_course_grpo_trainer_qwen3_4b_main.sh my_domain_task 192.168.1.10
```

The script uses:

- `task_name`: name of the domain task. By default, the input file is resolved as `/home/Qwen3/dataset/${date}/${task_name}.jsonl`.
- `master_ip`: Ray head node IP address. If omitted, the script uses its internal default.

Before launching, update the path variables in `MindSpeed-RL/examples/grpo/loop_course_grpo_trainer_qwen3_4b_main.sh`:

```bash
export MA_JOB_DIR=/home/Qwen3
export INPUT_JSONL=/path/to/domain_data.jsonl
export model_path=/path/to/Qwen3-4B-hf
export CKPT_LOAD_DIR=/path/to/Qwen3-4B-mcore
export CKPT_SAVE_DIR=/path/to/output_dir
```

Also verify:

- `TOTAL_ROUNDS`: number of EDCO curriculum rounds. The default is `5`.
- `samples_num`: number of selected samples per round. The default is `200`.
- `SOCKET_IFNAME`: network interface used by Ray. It defaults to `eth0` in the GRPO launcher.
- `NPUS_PER_NODE` and `NNODES`: resource settings in the GRPO launch script.

## Training Outputs

For each round, EDCO writes selected data, generated configs, logs, and checkpoints under `CKPT_SAVE_DIR`:

```text
<CKPT_SAVE_DIR>/
|-- data/
|   |-- selected_data_round1.jsonl
|   |-- selected_data_round1_with_entropy.jsonl
|   `-- round1/processed/
|-- models/
|   |-- round1_tuned_mc/
|   `-- round1_tuned_hf/
`-- iterative_training.log
```

`round<N>_tuned_mc` is used for GRPO training, while `round<N>_tuned_hf` is used by the entropy selector in the next round.

## Reward Verification

EDCO can use rule-based or model-based verification for reward computation. The main customization point is:

```text
MindSpeed-RL/mindspeed_rl/models/rule_verifier.py
```

The default training template enables rule reward through:

```yaml
rl_config:
  rule_reward: true
  verifier_function: ["base_acc"]
  verifier_weight: [1.0]
```

For a new domain, update the verifier so that it returns reward signals compatible with GRPO training. If you use a model judge, ensure that the judge output is parsed into a stable scalar reward.

## Checkpoint Conversion

EDCO alternates between two checkpoint formats:

- Hugging Face format for entropy-based selection with vLLM.
- MCore format for MindSpeed-RL training.

The Qwen3-4B loop calls:

```text
MindSpeed-LLM/examples/zyf/4B/ckpt_convert_qwen3_mcore2hf_loop.sh
```

If you adapt EDCO to another model size, update both the GRPO training config and the checkpoint conversion script.

## Citation

If this repository is useful for your work, please this work as:

```bibtex
@inproceedings{edco2026,
  title = {EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning},
  author = {Jing-Cheng Pang and Sun Liu and Chang Zhou and Xian Tang and Haichuan Ma and Kun Jiang and Jianlong Wang and Kai Zhang and Sijie Wu and Haoran Cai and Chenwei Wu and Xubin Li and Xin Chen},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026}
}
```

The complete proceedings metadata will be added when the official ICML entry is available.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details. The MindSpeed-LLM and MindSpeed-RL components retain their upstream license notices.
