# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
import time
from functools import partial

import ray
import torch

from megatron.training import get_args, get_timers, get_one_logger
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.training import append_to_progress_log, print_datetime, get_model
from megatron.training.utils import unwrap_model
from megatron.training.initialize import initialize_megatron
from mindspeed_llm.tasks.checkpoint.models import load_checkpoint
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, make_batch_generator
from mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional import split_dict_tensor_into_batches
from mindspeed_llm.tasks.posttrain.utils import compute_log_probs
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker import MegatronWorker
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.decorator import register, Dispatch
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.training.utils import get_tune_attention_mask

_TRAIN_START_TIME = time.time()


@ray.remote
class ReferenceWorker(MegatronWorker):
    """
    Ray ReferenceWorker
    """

    def __init__(self, config, role):
        super().__init__()

        self.config = config
        self.role = role

        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        initialize_megatron(role=self.role,
                            config=self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        self.reference = MegatronPPOReference()

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        output = self.reference.compute_log_prob(data=data)
        if output is not None:
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = output.to('cpu')
        torch.cuda.empty_cache()
        return output


class MegatronPPOReference(BaseTrainer):
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.args = get_args()
        self.timers = get_timers()

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
        print_rank_0('Time to initialize Megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')
        one_logger = get_one_logger()
        if one_logger:
            one_logger.log_metrics({
                'train_iterations_warmup': 5
            })

        self.timers('model-setup', log_level=0).start(barrier=True)

        self.model = get_model(self.model_provider, self.model_type, wrap_with_ddp=False)
        unwrapped_model = unwrap_model(self.model)
        if self.args.stage == "ray_online_dpo":
            self.args.micro_batch_size *= 2

        if self.args.load is not None or self.args.pretrained_checkpoint is not None:
            self.timers('load-checkpoint', log_level=0).start(barrier=True)
            self.args.iteration, self.args.num_floating_point_operations_so_far = load_checkpoint(
                self.model, None, None)
            self.timers('load-checkpoint').stop(barrier=True)
            self.timers.log(['load-checkpoint'])
        else:
            self.args.iteration = 0
            self.args.num_floating_point_operations_so_far = 0

        # get model without FP16 and/or DDP wrappers
        if self.args.iteration == 0 and len(unwrapped_model) == 1 \
                and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()

        self.timers('model-setup').stop()
        print_datetime('after model built')

    def compute_log_prob(self, data: DataProto):

        data.batch = data.batch.contiguous()

        for model_module in self.model:
            model_module.eval()

        data = data.to(next(self.model[0].parameters()).device)
        with torch.no_grad():
            output = self.forward_backward_batch(data, forward_only=True)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                ref_log_probs = torch.cat([out['ref_log_probs'] for out in output], dim=0)  # (bs, seq_size)
                ref_log_probs = ref_log_probs.to(torch.float32)
            else:
                ref_log_probs = None

        return ref_log_probs

    def forward_backward_batch(self, data, forward_only=False):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        args = get_args()
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)

        if data.meta_info.get('micro_batch_size', None) is not None:
            batch_size = data.meta_info['micro_batch_size']
        else:
            batch_size = args.micro_batch_size
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)

        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        forward_backward_func = get_forward_backward_func()

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.model))
        losses_reduced = forward_backward_func(
            forward_step_func=self.forward_step,
            data_iterator=batch_generator,
            model=self.model,
            num_microbatches=n_micro_batch,
            seq_length=seq_len,  # unused when variable_seq_lengths
            micro_batch_size=args.micro_batch_size,  # unused when variable_seq_lengths
            forward_only=forward_only
        )
        return losses_reduced

    def forward_step(self, batch_iter, model):
        input_ids, attention_mask, position_ids, response = self.get_batch(batch_iter)
        output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return output, partial(self.loss_func, response=response)

    def get_batch(self, batch_iter):
        batch = next(batch_iter)
        input_ids = batch['input_ids']
        attention_mask_1d = batch['attention_mask']
        position_ids = batch['position_ids']
        attention_mask = get_tune_attention_mask(attention_mask_1d)
        response = batch['responses']
        return input_ids, attention_mask, position_ids, response

    def loss_func(self, output, response):
        response_length = response.size(1)
        logits = output
        logits = logits[:, -response_length - 1:-1]
        log_probs = compute_log_probs(logits, response, per_token=True)[2]
        return 1.0, {'ref_log_probs': log_probs}
