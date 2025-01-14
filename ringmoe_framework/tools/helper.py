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
"""helper of ringmoe_framework"""
import sys

import aicc_tools as ac
import numpy as np
import yaml
import os
from mindspore.common import set_seed
from mindspore import context
from mindspore.ops.primitive import constexpr
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs


def build_context(args):
    """build context"""
    profile_cb = None
    if args.train_config.profile and args.use_parallel:
        cfts_1 = ac.CFTS(**args.aicc_config)
        profile_cb = cfts_1.profile_monitor(start_step=1, stop_step=5)
    context.set_context(mode=context.GRAPH_MODE)
    local_rank, device_num = ac.context_init(seed=args.seed, use_parallel=args.use_parallel,
                                             context_config=args.context, parallel_config=args.parallel)
    # context.set_auto_parallel_context(
    #     dataset_strategy=((device_num, 1, 1, 1), (device_num, 1, 1, 1), (device_num, 1), (device_num, 1)))

    context.set_context(max_device_memory="31GB")
    # context.set_context(save_graphs=2, save_graphs_path="/home/ma-user/modelarts/outputs/modelArts_output_0/")
    # comm_fusion_config = {
    #     "allreduce": {"mode": "size", "config": 256},
    #     "allgather": {"mode": "size", "config": 256},
    #     "reducescatter": {"mode": "size", "config": 256}
    # }
    comm_fusion_config = {
        "allreduce": {"mode": "size", "config": 32},
        "allgather": {"mode": "size", "config": 32},
        "reducescatter": {"mode": "size", "config": 32}
    }
    context.set_auto_parallel_context(comm_fusion=comm_fusion_config)

    ### 优化器并行，尝试降低显存
    parallel_optimizer_config = {"gradient_accumulation_shard": False, "parallel_optimizer_threshold": 64}
    context.set_auto_parallel_context(parallel_optimizer_config= parallel_optimizer_config)

    context.set_auto_parallel_context(pipeline_stages=args.parallel_config.pipeline_stage)
    # auto_parallel_context().set_enable_all_reduce_fusion(False)
    # auto_parallel_context().set_enable_all_gather_fusion(False)
    # auto_parallel_context().set_enable_reduce_scatter_fusion(False)

    set_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=False)
    _set_multi_subgraphs()

    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = ac.get_logger()
    args.logger.info("model config: {}".format(args))

    # init cfts
    if args.aicc_config.rank_id:
        cfts = ac.CFTS(**args.aicc_config)
    else:
        args.aicc_config.rank_id = local_rank
        cfts = ac.CFTS(**args.aicc_config)

    if args.callback.ckpt_config.obs_local_path:
        save_path = os.path.join(args.callback.ckpt_config.obs_local_path,args.callback.ckpt_config.prefix)
        strategy_save_path = os.path.join(save_path, 'strategy')
        strategy_save_path = os.path.join(strategy_save_path, str(args.local_rank))
        strategy_ckpt_save_file = os.path.join(strategy_save_path, f'strategy_{args.local_rank}.ckpt')
        context.set_auto_parallel_context(strategy_ckpt_save_file=strategy_ckpt_save_file)
    else:
        context.set_auto_parallel_context(strategy_ckpt_save_file=f'/cache/strategy_{args.local_rank}.ckpt')


    if args.parallel.get("strategy_ckpt_load_file"):
        args.parallel["strategy_ckpt_load_file"] = cfts.get_checkpoint(args.parallel.get("strategy_ckpt_load_file"))
        context.set_auto_parallel_context(strategy_ckpt_load_file=args.parallel["strategy_ckpt_load_file"])

    if args.train_config.profile and not args.use_parallel:
        cfts_2 = ac.CFTS(**args.aicc_config)
        profile_cb = cfts_2.profile_monitor(start_step=1, stop_step=5)
    return cfts, profile_cb


@constexpr
def check_modal_num(modal_num):
    if modal_num == 2:
        return True
    return False


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def parse_with_config(parser):
    """Parse With Config"""
    args = parser.parse_args()
    if args.config is not None:
        config_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = [np.prod(param.shape) for param in net.trainable_params()]
    return sum(total_params) // 1000000
