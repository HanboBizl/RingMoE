import mindspore.common.dtype as mstype
from mindspore import nn
# from mindspore.nn.transformer.moe import default_moe_config, calculate_expert_capacity, Router
##taoht##
from mindspore.parallel._transformer.moe import default_moe_config, calculate_expert_capacity, Router
from mindspore.parallel._transformer.op_parallel_config import MoEParallelConfig
# from mindspore.nn.transformer.op_parallel_config import default_moeparallel_config 
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .mlp import MLP

##taoht##
default_moeparallel_config = MoEParallelConfig()
class Moe(nn.Cell):
    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 modal_num=1,
                 use_dropout=False,
                 hidden_act='gelu',
                 weight_init='XavierUniform',
                 param_init_type=mstype.float32,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config):
        super(Moe, self).__init__()
        self.hidden_size = hidden_size
        self.modal_num = modal_num
        self.expert_dim = moe_config.expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.aux_loss_factor = moe_config.aux_loss_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.dp_group = parallel_config.data_parallel
        self.dp = parallel_config.data_parallel
        self.ep = parallel_config.expert_parallel

        self.ffn = MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            dropout_rate=dropout_rate,
            hidden_act=hidden_act,
            use_dropout=use_dropout,
            expert_num=self.expert_dim,
            weight_init=weight_init,
            param_init_type=param_init_type,
            parallel_config=parallel_config)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.transpose_2dim = P.Transpose().shard(((self.dp, 1),))
        self.transpose_2dim_ep = P.Transpose().shard(((self.ep, 1),))
        self.transpose_3dim = P.Transpose().shard(((self.dp, 1, 1),))
        self.transpose_4dim_ep = P.Transpose().shard(((self.ep, 1, 1, 1),))
        self.batch_mm = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.batch_mm2 = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.mul = P.Mul().shard(((), ()))
        self.router = Router(d_model=hidden_size, moe_config=moe_config, routing_policy=None,
                             training=True, parallel_config=parallel_config)
        self.cast = P.Cast()

    def construct(self, input_tensor):
        input_shape = F.shape(input_tensor)
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))
        bs_and_dmodel = self.shape(input_tensor)
        tokens_per_group = bs_and_dmodel[0] // self.dp_group
        input_tensor = self.reshape(input_tensor, (self.dp_group, tokens_per_group, self.hidden_size))

        expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group,
                                                    self.capacity_factor, self.expert_dim)
        # dispatch_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        # combine_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        dispatch_tensor, combine_tensor, aux_loss = self.router(input_tensor)

        # after transpose, input_tensor's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        input_tensor = self.transpose_3dim(input_tensor, (0, 2, 1))
        dispatch_tensor = self.reshape(dispatch_tensor, (self.dp_group, tokens_per_group,
                                                         self.expert_dim * expert_capacity))
        dispatch_tensor = self.cast(dispatch_tensor, F.dtype(input_tensor))
        # expert_input's shape: (self.dp_group, self.hidden_size, self.expert_dim * expert_capacity)
        expert_input = self.batch_mm(input_tensor, dispatch_tensor)
        expert_input = self.reshape(expert_input, (self.dp_group, self.hidden_size, self.expert_dim,
                                                   expert_capacity))
        # The following four ops are to implement transpose(expert_input, (2, 0, 3, 1)), for that a single transpose
        # has bad performance
        expert_input = self.reshape(expert_input, (self.dp_group * self.hidden_size,
                                                   self.expert_dim * expert_capacity))
        expert_input = self.transpose_2dim(expert_input, (1, 0))
        expert_input = self.reshape(expert_input, (self.expert_dim, expert_capacity, self.dp_group,
                                                   self.hidden_size))
        # expert_input's shape: (self.expert_dim, self.dp_group, expert_capacity, self.hidden_size)
        expert_input = self.transpose_4dim_ep(expert_input, (0, 2, 1, 3))
        expert_input = self.reshape(expert_input, (self.expert_dim * self.dp_group * expert_capacity,
                                                   self.hidden_size))

        # expert_output's shape: (self.expert_dim, self.dp_group*expert_capacity, self.hidden_size)
        expert_output = self.ffn(expert_input)
        expert_output = self.reshape(expert_output, (self.expert_dim, self.dp_group,
                                                     expert_capacity, self.hidden_size))
        # The following five ops are to implement transpose(expert_output, (1, 3, 0, 2)), for that a single transpose
        # has bad performance
        expert_output = self.reshape(expert_output, (self.expert_dim,
                                                     self.dp_group * expert_capacity * self.hidden_size))
        expert_output = self.transpose_2dim_ep(expert_output, (1, 0))
        expert_output = self.reshape(expert_output, (self.dp_group, expert_capacity,
                                                     self.hidden_size * self.expert_dim))
        expert_output = self.transpose_3dim(expert_output, (0, 2, 1))
        # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
        expert_output = self.reshape(expert_output, (self.dp_group, self.hidden_size, self.expert_dim,
                                                     expert_capacity))
        expert_output = self.reshape(expert_output, (self.dp_group, self.hidden_size,
                                                     self.expert_dim * expert_capacity))
        combine_tensor = self.reshape(combine_tensor, (self.dp_group, tokens_per_group,
                                                       self.expert_dim * expert_capacity))
        # combine_tensor's shape: (self.dp_group, self.expert_dim*expert_capacity, tokens_per_group)
        combine_tensor = self.transpose_3dim(combine_tensor, (0, 2, 1))
        combine_tensor = self.cast(combine_tensor, F.dtype(expert_output))

        # combined_output's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        combined_output = self.batch_mm2(expert_output, combine_tensor)
        # combined_output's shape: (self.dp_group, tokens_per_group, self.hidden_size)
        combined_output = self.transpose_3dim(combined_output, (0, 2, 1))
        combined_output = self.reshape(combined_output, (bs_and_dmodel[0], bs_and_dmodel[1]))
        combined_output = self.reshape(combined_output, input_shape)

        aux_loss = self.mul(self.aux_loss_factor, aux_loss)
        return combined_output, aux_loss
