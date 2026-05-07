# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import hydra

from mindspeed_llm.tasks.posttrain.launcher import get_trainer


@hydra.main(config_path='configs/rlxf', config_name='ppo_trainer_llama32_1b', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "True",
                         'TOKENIZERS_PARALLELISM': 'true',
                         'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    trainer = get_trainer(config.training.stage)(config)
    trainer.train()


if __name__ == '__main__':
    main()