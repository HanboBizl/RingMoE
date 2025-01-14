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
"""mlp of ringmo_framework"""
import mindspore.common.dtype as mstype
from mindspore import nn
# from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, _check_config
##taoht##
from mindspore.parallel._transformer.op_parallel_config import default_dpmp_config, _check_config
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .layers import Linear, Dropout

class MLP(nn.Cell):
    r"""MLP for ring-mo."""

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size=None,
                 out_features=None,
                 dropout_rate=0.,
                 hidden_act='gelu',
                 expert_num=1,
                 use_dropout=False,
                 modal_tag=False,
                 weight_init='XavierUniform',
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        ffn_hidden_size = ffn_hidden_size or hidden_size
        super(MLP, self).__init__()
        _check_config(parallel_config)
        mp = parallel_config.model_parallel
        if expert_num > 1:
            if modal_tag:
                ep = parallel_config.expert_parallel
                # ep = 2 * parallel_config.expert_parallel
            else:
                ep = parallel_config.expert_parallel
        else:
            ep = 1
        # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
        dp = int(parallel_config.data_parallel /ep)
        out_features = out_features or hidden_size
        # Project to ffn_hidden_size
        self.mapping = Linear(in_channels=hidden_size,
                              out_channels=ffn_hidden_size,
                              activation=hidden_act,
                              transpose_b=False,
                              expert_num=expert_num,
                              outer_batch=dp,
                              weight_init=weight_init,
                              param_init_type=param_init_type)

        if expert_num > 1:
            # self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
            #                    strategy_bias=((dp, ep, 1, mp), (mp,)),
            #                    strategy_activation=((dp, ep, 1, mp),))
            # 0407 expert_parallel
            self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                               strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)),
                               strategy_activation=((dp, ep, 1, mp),))
        else:
            self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                               strategy_bias=((dp, mp), (mp,)),
                               strategy_activation=((dp, mp),))
        # Project back to hidden_size
        self.projection = Linear(in_channels=ffn_hidden_size,
                                 out_channels=out_features,
                                 transpose_b=False,
                                 expert_num=expert_num,
                                 outer_batch=dp,
                                 weight_init=weight_init,
                                 param_init_type=param_init_type)
        if expert_num > 1:
            self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                  strategy_bias=((dp, ep, 1, 1), (1, ep, 1, 1)))
        else:
            self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                                  strategy_bias=((dp, 1), (1,)))
        # 0407 expert_parallel
        # if expert_num > 1:
        #     self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
        #                           strategy_bias=((dp, ep, 1, 1), (1, ep, 1, mp)))
        # else:
        #     self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
        #                           strategy_bias=((dp, 1), (1,)))
        self.projection.bias.parallel_optimizer = False
        self.dropout = Dropout(1 - dropout_rate)
        self.dropout.shard(((dp, 1),))
        self.dropout_3d = Dropout(1 - dropout_rate)
        self.dropout_3d.shard(((dp, 1, 1),))
        self.dropout_4d = Dropout(1 - dropout_rate)
        self.dropout_4d.shard(((dp, ep, 1, 1),))
        self.use_dropout = use_dropout
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)

        if self.use_dropout:
            if len(F.shape(hidden)) == 3:
                hidden = self.dropout_3d(hidden)
            elif len(F.shape(hidden)) == 2:
                hidden = self.dropout(hidden)
            else:
                hidden = self.dropout_4d(hidden)

        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        else:
            output = self.dropout_4d(output)
        return output
