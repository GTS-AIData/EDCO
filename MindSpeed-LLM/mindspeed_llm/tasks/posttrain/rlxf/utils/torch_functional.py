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
Contain small torch utilities
"""

from typing import List

import torch
import torch.distributed
from tensordict import TensorDict


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        elif mask_sum == 1:
            bessel_correction = mask_sum
        else:
            bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    if tensors.batch_size[0] % batch_size != 0:
        raise ValueError(f"Input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}. "
                        f"Please ensure that the input batch size is divisible by the split batch size.")
    return tensors.split(batch_size)
