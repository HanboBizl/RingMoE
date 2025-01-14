# Copyright 2021 Huawei Technologies Co., Ltd
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
"""parallel config"""

# from mindspore.nn.transformer.moe import default_moe_config, MoEConfig
# from mindspore.nn.transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

##taoht##
# from mindspore.parallel._transformer.moe import default_moe_config, MoEConfig
from mindspore.parallel._transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindspore import _checkparam as Validator

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)

class MoEConfig:
    r"""
        The configuration of MoE (Mixture of Expert).

        Args:
            expert_num (int): The number of experts employed. Default: 1
            capacity_factor (float): The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor (float): The factor is used to indicate how much the load balance loss (produced by the
                router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen (int): The number of experts is chosen by each token and it should not be larger
                than expert_num. Default: 1.
            expert_group_size (int): The number of tokens in each data parallel group. Default: ``None``.
                This parameter is effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            group_wise_a2a (bool): Whether to enable group-wise alltoall communication, which can reduce communication
                time by converting part of inter communication into intra communication. Default: ``False``.
                This parameter is effective only when model parallel > 1 and data_parallel equal to expert parallel.
            comp_comm_parallel (bool): Whether to enable ffn compute and communication parallel, which can reduce pure
                communicattion time by splitting and overlapping compute and communication. Default: ``False``.
            comp_comm_parallel_degree (int): The split number of compute and communication. The larger the numbers,
                the more overlap there will be but will consume more memory. Default: 2. This parameter is effective
                only when comp_comm_parallel enable.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self, expert_num=1, specific_expert_num=1, public_expert_num=1, cross_expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05, num_experts_chosen=1,
                 expert_group_size=None, group_wise_a2a=False, comp_comm_parallel=False, comp_comm_parallel_degree=2):
        Validator.check_positive_int(expert_num, "expert_num")
        Validator.check_positive_int(specific_expert_num, "specific_expert_num")
        Validator.check_positive_int(cross_expert_num, "cross_expert_num")
        Validator.check_positive_int(public_expert_num, "public_expert_num")
        Validator.check_positive_float(capacity_factor, "capacity_factor")
        Validator.check_positive_float(aux_loss_factor, "aux_loss_factor")
        Validator.check_positive_int(num_experts_chosen, "num_experts_chosen")
        Validator.check_bool(group_wise_a2a, "group_wise_a2a")
        Validator.check_bool(comp_comm_parallel, "comp_comm_parallel")
        Validator.check_positive_int(comp_comm_parallel_degree, "comp_comm_parallel_degree")
        if expert_group_size is not None:
            Validator.check_positive_int(expert_group_size, "expert_group_size")
        if capacity_factor < 1.0:
            raise ValueError(f"'capacity_factor' must be equal to or greater than 1.0, "
                             f"but got {capacity_factor}.")
        if aux_loss_factor >= 1.0:
            raise ValueError(f"'aux_loss_factor' must be less than 1.0, "
                             f"but got {aux_loss_factor}.")
        if num_experts_chosen > expert_num:
            raise ValueError(f"'num_experts_chosen' must not be larger than 'expert_num', "
                             f"but got {num_experts_chosen}.")
        self.expert_num = expert_num

        self.specific_expert_num = specific_expert_num
        self.cross_expert_num = cross_expert_num
        self.public_expert_num = public_expert_num

        self.capacity_factor = capacity_factor
        self.aux_loss_factor = aux_loss_factor
        self.num_experts_chosen = num_experts_chosen
        self.expert_group_size = expert_group_size
        self.group_wise_a2a = group_wise_a2a
        self.comp_comm_parallel = comp_comm_parallel
        self.comp_comm_parallel_degree = comp_comm_parallel_degree


default_moe_config = MoEConfig()

def build_parallel_config(config):
    """Build context config."""
    if config.moe_config:
        config.moe_config = MoEConfig(**config.moe_config)
    else:
        config.moe_config = default_moe_config
    if config.recompute_config:
        config.recompute_config = TransformerRecomputeConfig(**config.recompute_config)
    else:
        config.recompute_config = default_recompute_config
    if config.parallel_config:
        config.parallel_config = TransformerOpParallelConfig(recompute=config.recompute_config,
                                                             **config.parallel_config)
    else:
        config.parallel_config = default_parallel_config
