import time
import json
import os
from typing import Dict, Union, Iterable
from functools import partial

import ray
import torch
import torch_npu

import megatron
from megatron.training import print_rank_0
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.utils import get_model_config
from megatron.core import mpu
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt import GPTModel

from megatron.training.training import (
    print_datetime,
    get_one_logger,
    append_to_progress_log,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.checkpointing import save_checkpoint
from megatron.training.training import num_floating_point_operations
from megatron.training import get_args, initialize_megatron, get_timers

from mindspeed_llm.tasks.posttrain.utils import append_to_dict
from mindspeed_llm.training.utils import get_tune_attention_mask
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.orm.orm_model import GPTRewardModel
from mindspeed_llm.training.initialize import set_jit_fusion_options

from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.decorator import register, Dispatch
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker import MegatronWorker
from mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional import masked_mean, split_dict_tensor_into_batches
from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, make_batch_generator
from mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional import clip_by_value

_TRAIN_START_TIME = time.time()


@ray.remote
class CriticWorker(MegatronWorker):
    def __init__(self, config, role):
        """
        """
        super().__init__()

        self.config = config
        self.role = role
        self.IGNORE_INDEX = -100
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        initialize_megatron(role=self.role,
                            config=self.config)

        self.args = get_args()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        self.critic = MegatronPPOCritic()

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={'values': values})
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        dataloader = self.critic.make_minibatch_iterator(data)
        metrics = self.critic.update_critic(dataloader=dataloader)
        output = DataProto(batch=None, meta_info={'metrics': metrics})
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, iteration):
        self.critic.save_checkpoint(iteration)


class MegatronPPOCritic(BaseTrainer):
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.args = get_args()
        self.timers = get_timers()
        self.num_floating_point_operations_so_far = 0

        if self.args.log_progress:
            append_to_progress_log("Starting job")
        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()
        # Adjust the startup time, so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor(
            [_TRAIN_START_TIME],
            dtype=torch.float,
            device='cuda'
        )
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        one_logger = get_one_logger()
        if one_logger:
            one_logger.log_metrics({
                'train_iterations_warmup': 5
            })

        from megatron.training.training import setup_model_and_optimizer
        # Model, optimizer, and learning rate.
        self.timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            self.model_provider, self.model_type)

        self.timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                       'scheduler are built')
        model_config = get_model_config(model[0])

        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.model_config = model_config
        self.process_non_loss_data_func = None

    def save_checkpoint(self, iteration):
        save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                        self.num_floating_point_operations_so_far)


    @staticmethod
    def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
        """Builds the model.

        Currently supports only the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
            Defaults to True.

        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if not args.use_mcore_models:
            raise ValueError("Reward model training currently supports mcore only.")

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                    args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTRewardModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            post_layer_norm=not args.no_post_layer_norm,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

        return model

    def critic_data_padding(self, data: DataProto) -> DataProto:
        if 'response_mask' in data.batch.keys():
            return data

        prompt_length = data.batch['prompts'].shape[1]
        response_mask = data.batch['attention_mask']
        response_mask[..., :prompt_length] = 0
        data.batch['response_mask'] = response_mask

        return data

    def get_batch(self, data_iterator):
        self.timers('batch-generator', log_level=2).start()
        batch = next(data_iterator)
        input_ids = batch["input_ids"]
        attention_mask_1d = batch["attention_mask"]
        attention_mask = get_tune_attention_mask(attention_mask_1d)
        position_ids = batch["position_ids"]

        return batch, input_ids, attention_mask, position_ids

    def forward_backward_batch(self, data_proto: DataProto, forward_only=False):
        data_proto.batch = data_proto.batch.contiguous()
        args = get_args()
        data = data_proto.batch

        forward_batch_size = data["input_ids"].shape[0]
        forward_num_microbatches = forward_batch_size // args.micro_batch_size

        batches = split_dict_tensor_into_batches(data, batch_size=args.micro_batch_size)
        data_iterator = make_batch_generator(batches, vpp_size=len(self.model))

        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=self.forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=forward_num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            collect_non_loss_data=forward_only,
            forward_only=forward_only)

        return losses_reduced

    def compute_values(self, data: DataProto):
        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.size(1)

        for model_module in self.model:
            model_module.eval()

        with torch.no_grad():
            output = self.forward_backward_batch(data, forward_only=True)

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                values = torch.cat(output, dim=0).squeeze(-1)
                values = values.to(torch.float32)
            else:
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            values = values * attention_mask
            values = values[:, -response_length - 1:-1]
            values = values.contiguous()

        self.args.consumed_train_samples += self.args.global_batch_size
        self.num_floating_point_operations_so_far += num_floating_point_operations(self.args, self.args.global_batch_size)
        torch.cuda.empty_cache()
        return values

    def update_critic(self, dataloader: Iterable[DataProto]):
        metrics = {}
        for model_module in self.model:
            model_module.train()

        for data in dataloader:
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()

            self.optimizer.zero_grad()

            metric_micro_batch = self.forward_backward_batch(data, forward_only=False)

            # Empty unused memory.
            if self.args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

            # # Update learning rate.
            if update_successful:
                increment = self.args.critic_mini_batch_size
                self.opt_param_scheduler.step(increment=increment)

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)

        torch.cuda.empty_cache()
        return metrics

    def make_minibatch_iterator(self, data: DataProto):
        select_keys = data.batch.keys()
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.args.critic_mini_batch_size,
                                  epochs=self.args.critic_update_epochs,
                                  )

    def loss_func(self, data, output_tensor, non_loss_data=False):
        if non_loss_data:
            return output_tensor

        responses = data['responses']
        response_length = responses.size(1)
        attention_mask = data['attention_mask']
        eos_mask = attention_mask[:, -response_length:]
        eos_p1_index = torch.min(torch.cumsum(eos_mask, dim=-1)[:, -1],
                        torch.tensor(eos_mask.shape[1], device=eos_mask.device))
        eos_mask[:, eos_p1_index] = 1

        cliprange_value = self.args.cliprange_value
        curr_values = output_tensor.squeeze(-1)
        curr_values = curr_values[:, -response_length - 1:-1]
        curr_values = torch.masked_fill(curr_values, ~eos_mask, 0)

        returns = data['returns']

        if cliprange_value > 0.0:
            prev_values = data['values']
            vpredclipped = clip_by_value(curr_values, prev_values - cliprange_value, prev_values + cliprange_value)
            vf_losses1 = (vpredclipped - returns) ** 2
        else:
            vf_losses1 = torch.tensor(0.0).to(curr_values.device)

        vf_losses2 = (curr_values - returns) ** 2

        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)

        stats = {
            'critic/vf_loss': vf_loss.detach().item(),
            'critic/vf_clipfrac': vf_clipfrac.detach().item(),
            'critic/vpred_mean': masked_mean(curr_values, eos_mask).detach().item(),
        }

        return vf_loss, stats

    def forward_step(self, data_iterator, model):
        batch, input_ids, attention_mask, position_ids = self.get_batch(data_iterator)
        output_tensor = model(input_ids, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, batch)
