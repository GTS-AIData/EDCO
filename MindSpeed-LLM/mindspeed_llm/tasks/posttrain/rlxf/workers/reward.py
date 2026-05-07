# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
import time
from functools import partial
import re

import ray
import torch
from transformers import AutoTokenizer

from megatron.training import print_rank_0
from megatron.training import get_args, get_timers, get_one_logger
from megatron.training.utils import unwrap_model
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import append_to_progress_log, print_datetime, get_model
from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.core.pipeline_parallel import get_forward_backward_func

from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.training.utils import get_tune_attention_mask
from mindspeed_llm.tasks.posttrain.orm import ORMTrainer
from mindspeed_llm.tasks.posttrain.rlxf.workers.actor_train_infer import pad_to_tensor_dict, \
    generate_attention_mask, generate_position_ids_from_attention_mask
from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto
from mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional import split_dict_tensor_into_batches
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker import MegatronWorker
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.decorator import register, Dispatch


_TRAIN_START_TIME = time.time()


@ray.remote
class RewardWorker(MegatronWorker):
    """
    Ray RewardWorker
    """
    def __init__(self, config, role):
        super().__init__()

        self.config = config
        self.role = role
        self.rm = None

        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        initialize_megatron(role=self.role,
                            config=self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        self.rm = MegatronPPORM()

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')
        output = self.rm.compute_rm_score(data=data)
        output = DataProto.from_dict(tensors={'rm_scores': output})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output


class MegatronPPORM(ORMTrainer):
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

        if self.args.stage == "ray_online_dpo":
            self.args.micro_batch_size *= 2
        self.timers('model-setup', log_level=0).start(barrier=True)

        model = get_model(self.model_provider, self.model_type, wrap_with_ddp=False)
        unwrapped_model = unwrap_model(model)

        if self.args.load is not None or self.args.pretrained_checkpoint is not None:
            self.timers('load-checkpoint', log_level=0).start(barrier=True)
            self.args.iteration, self.args.num_floating_point_operations_so_far = load_checkpoint(
                model, None, None, strict=True)
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

        self.timers('model-and-optimizer-setup').stop()
        print_datetime('after model built')
        config = get_model_config(model[0])

        # Print setup timing.
        self.train_args = [self.forward_step, model, config]
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path)

    def compute_rm_score(self, data: DataProto):
        args = get_args()
        forward_step_func, model, config = self.train_args
        prompt_lens = data.batch["prompts"].size(1)

        for model_module in model:
            model_module.eval()

        with torch.no_grad():
            batches = split_dict_tensor_into_batches(data.batch, batch_size=args.micro_batch_size)
            n_micro_batch = len(batches)

            batch_generator = iter(batches)
            forward_backward_func = get_forward_backward_func()

            rm_score = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=batch_generator,
                model=model,
                num_microbatches=n_micro_batch,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                collect_non_loss_data=True,
                forward_only=True
            )

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()
        if mpu.is_pipeline_last_stage():
            rm_score = torch.cat(rm_score, dim=0).squeeze(-1)  # (bs, seq_size)
            rm_score = rm_score.to(torch.float32)
            rm_score = rm_score[:, prompt_lens:]
        else:
            rm_score = torch.zeros(1)

        return rm_score

    def forward_step(self, data_iterator, model):
        """ReWardModel forward step to calculate rm scores.

        Args:
            data_iterator : Data iterator which wait to get input ids from Queue generated in Actor Server
            model (GPTModel): The GPT Model
        """
        self.timers('batch-generator', log_level=2).start()
        input_ids, attention_mask, position_ids = self._get_tokens(data_iterator)
        self.timers('batch-generator').stop()

        scores = model(input_ids, position_ids, attention_mask)

        return scores, self.loss_func

    def loss_func(self, scores: torch.Tensor, non_loss_data=False):
        return scores

    def _get_tokens(self, data_iterator):
        self.timers('batch-generator', log_level=2).start()
        batch = next(data_iterator)

        if self.args.extract_content_for_reward:
            str_responses = tokenizer.batch_decode(batch["responses"])
            pattern = r'<answer>(.*?)</answer>'
            contents = []
            for str_response in str_responses:
                first_pad_position = str_response.find(tokenizer.pad_token)
                if first_pad_position != -1:
                    str_response = str_response[:first_pad_position]
                within_answer = re.findall(pattern, str_response)
                if within_answer:
                    content = within_answer[0]
                else:
                    content = str_response
                contents.append(tokenizer.encode(content))

            responses_ori_length, responses_pad_length = pad_to_tensor_dict(
                contents,
                pad_multi_of=self.args.pad_to_multiple_of
            )

            prompts = batch["prompts"]
            prompts_pad_length = torch.LongTensor([len(prompts[0])]).cuda()
            pad_token_id = tokenizer.pad_token_id
            prompts_ori_length = [len(prompts[i]) - (prompts[i] == pad_token_id).sum().item() for i in range(len(prompts))]
            prompts = prompts.cpu().numpy().tolist()

            input_ids = [prompt + response for prompt, response in zip(prompts, contents)]
            attention_mask = generate_attention_mask(input_ids, prompts_ori_length, prompts_pad_length,
                                                     responses_ori_length, responses_pad_length)
            position_ids = generate_position_ids_from_attention_mask(input_ids, prompts_ori_length, prompts_pad_length)

            device = batch["input_ids"].device
            input_ids = torch.tensor(input_ids).long().to(device)
            attention_mask_1d = torch.tensor(attention_mask).long().to(device)
            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = torch.tensor(position_ids).long().to(device)

        else:
            input_ids = batch["input_ids"]
            attention_mask_1d = batch["attention_mask"]
            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = batch["position_ids"]

        return input_ids, attention_mask, position_ids
