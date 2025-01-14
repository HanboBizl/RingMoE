# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Operations for clipping tensors to min/max values."""
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr

from ..models.layers.layers import Identity


get_square_sum = C.MultitypeFuncGraph("get_square_sum")
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


@apply_global_norm.register("Bool", "Tensor", "Tensor", "Tensor")
def _apply_global_norm(enable_grad_fp16, clip_norm, global_norm, grad):
    if enable_grad_fp16:
        grad = P.Cast()(grad * clip_norm / global_norm, mstype.float16)
    else:
        grad = grad * clip_norm / global_norm
    return grad


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, parallel_config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_semi_parallel = context.get_auto_parallel_context("parallel_mode") == ParallelMode.SEMI_AUTO_PARALLEL
        self.group_size = os.getenv("RANK_SIZE", 1)
        self.merge_op = P.AllReduce() if self.is_semi_parallel else Identity
        if self.is_data_parallel:
            self.merge_op = P.identity()
        else:
            self.merge_op = P.AllReduce()
        if self.is_data_parallel:
            self.allreduce_group_size = (1,) * len(params)
        else:
            self.allreduce_group_size = (data_parallel * 1.0,) * len(params)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        global_norms = F.sqrt(self.merge_op(square_reduce_sum))
        return grads, global_norms

    def _get_scale_for_gradient_norm(self, params):
        allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name:
                allreduce_group_size = allreduce_group_size + (1.0,)
            elif "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
            else:
                allreduce_group_size = allreduce_group_size + (self.parallel_config.data_parallel * 1.0 * 1.0,)
        return allreduce_group_size


class ClipByGlobalNorm(nn.Cell):
    """Clip grads by global norm."""

    def __init__(self, params, data_parallel=1, param_init_type=mstype.float16, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params, parallel_config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        if param_init_type == mstype.float16:
            self.enable_grad_fp16 = True
        else:
            self.enable_grad_fp16 = False

    def construct(self, grads):
        """Clip grads by global norm construct"""
        grads, global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.enable_grad_fp16, self.clip_norm, global_norm), grads)
        return grads, global_norm_value
