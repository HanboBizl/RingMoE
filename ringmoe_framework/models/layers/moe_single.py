import mindspore.common.dtype as mstype
from mindspore import nn
import numpy as np
from mindspore.nn.layer import Dense
from mindspore.common.tensor import Tensor
# from mindspore.nn.transformer.moe import default_moe_config, calculate_expert_capacity, Router
##taoht##
from mindspore.parallel._transformer.moe import default_moe_config, calculate_expert_capacity, Router
from mindspore.parallel._transformer.op_parallel_config import MoEParallelConfig
# from mindspore.nn.transformer.op_parallel_config import default_moeparallel_config
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from .mlp import MLP

class Router(nn.Cell):
    r"""
        A router backbone used to calculate logits of each token, which should be cascaded by router implementations
        mapping tokens to experts.
        when moe_config.num_experts_chosen = 1, use top1 routing;
        when moe_config.num_experts_chosen > 1, use topk routing

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            routing_policy: The policy of mapping tokens to experts. Default: topkRouter
            training (bool): The value indicating whether is in training phase.
            parallel_config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 specific_expert_num=None,
                 routing_policy=None,
                 training=True,
                 parallel_config=None):
        super(Router, self).__init__()
        dp = parallel_config.data_parallel
        self.d_model = d_model
        self.expert_dim = moe_config.expert_num
        self.specific_expert_num = specific_expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.training = training
        self.routing_policy = routing_policy
        self.noisy_policy = None  # candidate: ["jitter", "rsample", "None"]
        self.noisy_epsilon = 1e-2
        self.noise = Tensor(np.random.uniform(1 - self.noisy_epsilon, 1 + self.noisy_epsilon, (d_model,)))
        if specific_expert_num:
            self.dense = Dense(in_channels=self.d_model, out_channels=self.specific_expert_num, has_bias=False)
        else:
            self.dense = Dense(in_channels=self.d_model, out_channels=self.expert_dim, has_bias=False)

        self.router = routing_policy
        if self.routing_policy is None:
            self.router = TopkRouter(d_model=d_model, moe_config=moe_config, training=training,specific_expert_num=specific_expert_num,
                                     parallel_config=parallel_config)

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.dense.matmul.shard(((dp, 1), (1, 1)))
            self.mul = P.Mul()
            self.cast = P.Cast()
        else:
            self.dense.matmul.shard(((dp, 1), (1, 1)))
            self.mul = P.Mul().shard(((dp, 1, 1), (dp,)))
            self.cast = P.Cast()

    def construct(self, input_tensor):
        input_tensor = self.cast(input_tensor, mstype.float32)
        if self.noisy_policy == "jitter" and self.training:
            # Here, we temporarily implement the multiplicative jitter this way,
            # for the lack of UniforReal parallel operator.
            input_tensor = self.mul(input_tensor, self.noise)

        router_logits = self.dense(input_tensor)
        return self.router(router_logits)


class TopkRouter(nn.Cell):
    r"""
        A router implementation which maps each tokens to the topk expert.

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            training (bool): The value indicating whether is in training phase.
            config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim, expert\_capacity)`,
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim, expert\_capacity)`,
            Tensor of shape :math:`(1)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 specific_expert_num=None,
                 training=True,
                 parallel_config=None):
        super(TopkRouter, self).__init__()
        dp = parallel_config.data_parallel
        self.d_model = d_model
        self.expert_dim = moe_config.expert_num
        self.specific_expert_num = specific_expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.training = training
        self.dp_group = dp
        self.noisy_policy = None
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.num_experts_chosen = moe_config.num_experts_chosen

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.softmax = P.Softmax(axis=-1)
            self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False)
            self.onehot = P.OneHot()
            self.onehot2 = P.OneHot()
            self.onehot3 = P.OneHot()

            self.reduce_mean = P.ReduceMean(keep_dims=False)
            self.reduce_mean2 = P.ReduceMean(keep_dims=False)
            self.reduce_mean3 = P.ReduceMean(keep_dims=False)
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.mul3 = P.Mul()
            self.mul4 = P.Mul()
            self.mul5 = P.Mul()
            self.mul6 = P.Mul()
            self.mul7 = P.Mul()
            self.mul8 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul9 = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.not_equal = P.NotEqual()
            self.div1 = P.RealDiv()
            self.div2 = P.RealDiv()
            self.add = P.Add()
            self.add1 = P.Add()
            self.add2 = P.Add()
            self.add3 = P.Add()
            self.add4 = P.Add()
            self.sub = P.Sub()

            self.cumsum = P.CumSum(exclusive=True)
            self.less = P.Less()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.reduce_sum_keep = P.ReduceSum(keep_dims=True)
            self.reduce_sum_keep2 = P.ReduceSum(keep_dims=True)
            self.expand = P.ExpandDims()
            self.expand2 = P.ExpandDims()
            self.add_scala = P.Add()
            self.init_loss = Tensor(0.0, mstype.float32)
        else:
            self.softmax = P.Softmax(axis=-1).shard(((dp, 1, 1,),))
            self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False).shard(((dp, 1, 1),))
            self.onehot = P.OneHot().shard(((dp, 1, 1), (), ()))
            self.onehot2 = P.OneHot().shard(((dp, 1, 1), (), ()))
            self.onehot3 = P.OneHot().shard(((dp, 1, 1, 1), (), ()))

            self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
            self.reduce_mean2 = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
            self.reduce_mean3 = P.ReduceMean(keep_dims=False).shard(((dp, 1),))
            self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
            self.mul2 = P.Mul().shard(((), ()))
            self.mul3 = P.Mul().shard(((), ()))
            self.mul4 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul5 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul6 = P.Mul().shard(((dp, 1), (dp, 1)))
            self.mul7 = P.Mul().shard(((dp, 1), (dp, 1)))
            self.mul8 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
            self.mul9 = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.not_equal = P.NotEqual().shard(((dp, 1, 1, 1), ()))
            self.div1 = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
            self.div2 = P.RealDiv().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
            self.add1 = P.Add().shard(((dp, 1, 1), ()))
            self.add2 = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            self.add3 = P.Add().shard(((dp, 1), (dp, 1)))
            self.add4 = P.Add().shard(((dp, 1, 1, 1), ()))
            self.sub = P.Sub().shard(((), (dp, 1, 1)))

            self.cumsum = P.CumSum(exclusive=True).shard(((dp, 1, 1),))
            self.less = P.Less().shard(((dp, 1, 1), ()))
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((dp, 1, 1),))
            self.reduce_sum_keep = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1),))
            self.reduce_sum_keep2 = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1, 1),))
            self.expand = P.ExpandDims().shard(((dp, 1),))
            self.expand2 = P.ExpandDims().shard(((dp, 1, 1),))
            self.add_scala = P.Add().shard(((), ()))
            self.init_loss = Tensor(0.0, mstype.float32)

    def construct(self, router_logits):
        router_logits_shape = self.shape(router_logits)
        router_logits = self.reshape(router_logits, (-1, router_logits_shape[-1]))
        logits_shape = self.shape(router_logits)
        tokens_per_group = logits_shape[0] // self.dp_group
        if self.specific_expert_num:
            expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group, self.capacity_factor,
                                                        self.specific_expert_num)
            router_logits = self.reshape(router_logits, (self.dp_group, tokens_per_group, self.specific_expert_num))
        else:
            expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group, self.capacity_factor,
                                                        self.expert_dim)
            router_logits = self.reshape(router_logits, (self.dp_group, tokens_per_group, self.expert_dim))

        accum_expert_mask = 0
        accum_expert_gate = 0
        loss = self.init_loss
        mask_count = 0
        accum_combine_tensor = 0
        # Probabilities for each token of what expert is should be sent to
        router_prob = self.softmax(router_logits)

        for expert_chosen_index in range(self.num_experts_chosen):
            # for each token, set the router_prob of the selected experts to zero
            router_prob = self.mul4(router_prob, self.sub(self.on_value, accum_expert_mask))
            # shape is : (dp_group, tokens_per_group)
            expert_index, expert_gate = self.argmax(router_prob)
            # expert_mask's shape: (dp_group, tokens_per_group, self.expert_dim)
            if self.specific_expert_num:
                expert_mask = self.onehot(expert_index, self.specific_expert_num, self.on_value, self.off_value)
            else:
                expert_mask = self.onehot(expert_index, self.expert_dim, self.on_value, self.off_value)
            # renormalize the rest prob to be of sum 1
            router_prob_normal = self.div1(router_prob, self.add1(self.reduce_sum_keep(router_prob, -1), 1e-9))

            # the balance loss is computed at each routing step
            loss = self.add_scala(loss, self._auxiliary_loss(expert_mask, router_prob_normal))

            output = self._maskout_overflowed_tokens(expert_mask, expert_capacity, expert_gate,
                                                     mask_count, expert_chosen_index)
            expert_mask, expert_gate, expert_mask_flat, position_in_expert = output[0], output[1], output[2], output[3]
            accum_expert_mask = self.add(accum_expert_mask, expert_mask)
            accum_expert_gate = self.add3(accum_expert_gate, expert_gate)
            mask_count = self.add(mask_count, self.reduce_sum_keep(expert_mask, 1))

            # combine_tensor's shape: (dp_group, tokens_per_group)
            combine_tensor = self.mul7(expert_gate, expert_mask_flat)
            # combine_tensor's shape: (dp_group, tokens_per_group, self.expert_dim)
            if self.specific_expert_num:
                combine_tensor = self.mul8(self.expand(combine_tensor, -1),
                                           self.onehot2(expert_index, self.specific_expert_num, self.on_value, self.off_value))
            else:
                combine_tensor = self.mul8(self.expand(combine_tensor, -1),
                                           self.onehot2(expert_index, self.expert_dim, self.on_value, self.off_value))
            # combine_tensor's shape: (dp_group, tokens_per_group, self.expert_dim, self.expert_capacity)
            combine_tensor = self.mul9(self.expand2(combine_tensor, -1),
                                       self.onehot3(self.cast(position_in_expert, mstype.int32), expert_capacity,
                                                    self.on_value, self.off_value))
            accum_combine_tensor = self.add2(accum_combine_tensor, combine_tensor)

        # expert weights normalization
        combine_tensor_sum = self.reduce_sum_keep2(self.reduce_sum_keep2(accum_combine_tensor, -1), -2)
        accum_combine_tensor = self.div2(accum_combine_tensor, self.add4(combine_tensor_sum, 1e-9))
        # dispatch_tensor is of boolean type. Here, using NotEqual instead of Cast, for that 'Cast to bool' has
        # bad performance
        dispatch_tensor = self.not_equal(accum_combine_tensor, 0.0)
        return dispatch_tensor, accum_combine_tensor, loss

    def _auxiliary_loss(self, expert_mask, router_prob):
        """
        Computing the load balance loss.
        """
        # density_1's shape: (dp_group, self.expert_dim)
        density_1 = self.reduce_mean(expert_mask, 1)
        # density_1_proxy's shape: (dp_group, self.expert_dim)
        density_1_proxy = self.reduce_mean2(router_prob, 1)
        loss = self.mul(density_1, density_1_proxy)
        loss = self.reduce_mean3(loss)
        loss = self.mul3(self.mul2(loss, self.expert_dim), self.expert_dim)
        return loss

    def _maskout_overflowed_tokens(self, expert_mask, expert_capacity, expert_gate, last_num, expert_chosen_index):
        """
        Keeping only the tokens that fit within expert_capacity.
        """
        cumsum = self.cumsum(expert_mask, 1)
        if expert_chosen_index > 0:
            cumsum = self.add(cumsum, last_num)
        # position_in_expert's shape: (dp_group, tokens_per_group, self.expert_dim)
        position_in_expert = self.mul4(cumsum, expert_mask)
        less_result = self.less(position_in_expert, expert_capacity)
        # expert_mask's shape: (dp_group, tokens_per_group, self.expert_dim)
        expert_mask = self.mul5(less_result, expert_mask)
        # expert_mask_flat's shape: (dp_group, tokens_per_group)
        expert_mask_flat = self.reduce_sum(expert_mask, -1)

        # Mask out the experts that have overflowed the expert_capacity.
        # expert_gate's shape: (dp_group, tokens_per_group)
        expert_gate = self.mul6(expert_gate, expert_mask_flat)
        output = (expert_mask, expert_gate, expert_mask_flat, position_in_expert)
        return output

##taoht##
default_moeparallel_config = MoEParallelConfig()
class Moe_single(nn.Cell):
    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 modal_num =1,
                 use_dropout=False,
                 hidden_act='gelu',
                 weight_init='XavierUniform',
                 param_init_type=mstype.float32,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config):
        super(Moe_single, self).__init__()
        self.hidden_size = hidden_size

        self.modal_num = modal_num
        self.expert_dim = moe_config.expert_num
        self.specific_expert_num = moe_config.specific_expert_num
        self.public_expert_num = moe_config.public_expert_num
        self.cross_expert_num = moe_config.cross_expert_num

        self.capacity_factor = moe_config.capacity_factor
        self.aux_loss_factor = moe_config.aux_loss_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.dp_group = parallel_config.data_parallel
        self.dp = parallel_config.data_parallel
        self.ep = parallel_config.expert_parallel
        self.special_ffn = MLP(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                dropout_rate=dropout_rate,
                hidden_act=hidden_act,
                use_dropout=use_dropout,
                modal_tag=False,
                expert_num=self.specific_expert_num,
                weight_init=weight_init,
                param_init_type=param_init_type,
                parallel_config=parallel_config)
        self.special_router = Router(d_model=hidden_size, moe_config=moe_config, specific_expert_num=self.specific_expert_num, routing_policy=None,
                                 training=True, parallel_config=parallel_config)

        self.cross_ffn = MLP(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                dropout_rate=dropout_rate,
                hidden_act=hidden_act,
                use_dropout=use_dropout,
                modal_tag = True,
                expert_num=self.cross_expert_num,
                weight_init=weight_init,
                param_init_type=param_init_type,
                parallel_config=parallel_config)
        self.cross_router = Router(d_model=hidden_size, moe_config=moe_config, specific_expert_num=self.cross_expert_num,
                        routing_policy=None,
                        training=True, parallel_config=parallel_config)

        self.public_ffn = MLP(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size*4,
                dropout_rate=dropout_rate,
                hidden_act=hidden_act,
                use_dropout=use_dropout,
                modal_tag=False,
                expert_num=self.public_expert_num,
                weight_init=weight_init,
                param_init_type=param_init_type,
                parallel_config=parallel_config)

        self.split_3d = P.Split(axis=0,output_num=self.modal_num).shard(((1, 1, 1),))
        self.cat_3d = P.Concat(axis=0).shard(((1, 1, 1), (1, 1, 1)))
        self.add_3d = P.Add().shard(((1, 1, 1), (1, 1, 1)))
        self.add = P.Add()
        self.div = P.Div()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.transpose_2dim = P.Transpose().shard(((self.dp, 1),))
        self.transpose_2dim_ep = P.Transpose().shard(((self.ep, 1),))
        self.transpose_2dim_ep_cross = P.Transpose().shard(((self.ep, 1),))
        # self.transpose_2dim_ep_cross = P.Transpose().shard(((self.ep*2, 1),))
        self.transpose_3dim = P.Transpose().shard(((self.dp, 1, 1),))
        self.transpose_4dim_ep = P.Transpose().shard(((self.ep, 1, 1, 1),))
        self.transpose_4dim_ep_cross = P.Transpose().shard(((self.ep, 1, 1, 1),))
        # self.transpose_4dim_ep_cross = P.Transpose().shard(((self.ep*2, 1, 1, 1),))
        self.batch_mm = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.batch_mm2 = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.mul = P.Mul().shard(((), ()))
        # self.router = Router(d_model=hidden_size, moe_config=moe_config, routing_policy=None,
        #                      training=True, parallel_config=parallel_config)
        self.cast = P.Cast()


    def _construct(self,router, ffn, input_tensor, expert_num, transpose_2dim_ep, transpose_4dim_ep):
        input_shape = F.shape(input_tensor)
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))
        bs_and_dmodel = self.shape(input_tensor)
        tokens_per_group = bs_and_dmodel[0] // self.dp_group
        input_tensor = self.reshape(input_tensor, (self.dp_group, tokens_per_group, self.hidden_size))

        expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group,
                                                    self.capacity_factor, expert_num)

        # dispatch_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        # combine_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        dispatch_tensor, combine_tensor, aux_loss = router(input_tensor)

        # after transpose, input_tensor's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        input_tensor = self.transpose_3dim(input_tensor, (0, 2, 1))
        dispatch_tensor = self.reshape(dispatch_tensor, (self.dp_group, tokens_per_group,
                                                         expert_num * expert_capacity))
        dispatch_tensor = self.cast(dispatch_tensor, F.dtype(input_tensor))
        # expert_input's shape: (self.dp_group, self.hidden_size, expert_num * expert_capacity)
        expert_input = self.batch_mm(input_tensor, dispatch_tensor)
        expert_input = self.reshape(expert_input, (self.dp_group, self.hidden_size, expert_num,
                                                   expert_capacity))
        # The following four ops are to implement transpose(expert_input, (2, 0, 3, 1)), for that a single transpose
        # has bad performance
        expert_input = self.reshape(expert_input, (self.dp_group * self.hidden_size,
                                                   expert_num * expert_capacity))
        expert_input = self.transpose_2dim(expert_input, (1, 0))
        expert_input = self.reshape(expert_input, (expert_num, expert_capacity, self.dp_group,
                                                   self.hidden_size))
        # expert_input's shape: (expert_num, self.dp_group, expert_capacity, self.hidden_size)

        expert_input = transpose_4dim_ep(expert_input, (0, 2, 1, 3))
        # expert_input = self.transpose_4dim_ep(expert_input, (0, 2, 1, 3))

        expert_input = self.reshape(expert_input, (expert_num * self.dp_group * expert_capacity,
                                                   self.hidden_size))

        # expert_output's shape: (expert_num, self.dp_group*expert_capacity, self.hidden_size)
        expert_output = ffn(expert_input)
        expert_output = self.reshape(expert_output, (expert_num, self.dp_group,
                                                     expert_capacity, self.hidden_size))
        # The following five ops are to implement transpose(expert_output, (1, 3, 0, 2)), for that a single transpose
        # has bad performance
        expert_output = self.reshape(expert_output, (expert_num,
                                                     self.dp_group * expert_capacity * self.hidden_size))

        expert_output = transpose_2dim_ep(expert_output, (1, 0))
        # expert_output = self.transpose_2dim_ep(expert_output, (1, 0))

        expert_output = self.reshape(expert_output, (self.dp_group, expert_capacity,
                                                     self.hidden_size * expert_num))
        expert_output = self.transpose_3dim(expert_output, (0, 2, 1))
        # expert_output's shape: (self.dp_group, self.hidden_size, expert_num, expert_capacity)
        expert_output = self.reshape(expert_output, (self.dp_group, self.hidden_size, expert_num,
                                                     expert_capacity))
        expert_output = self.reshape(expert_output, (self.dp_group, self.hidden_size,
                                                     expert_num * expert_capacity))
        combine_tensor = self.reshape(combine_tensor, (self.dp_group, tokens_per_group,
                                                       expert_num * expert_capacity))
        # combine_tensor's shape: (self.dp_group, expert_num*expert_capacity, tokens_per_group)
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

    def construct(self, modal_input):
        cross_out, cross_aux_loss = self._construct(self.cross_router,self.cross_ffn,modal_input,self.cross_expert_num, self.transpose_2dim_ep_cross, self.transpose_4dim_ep_cross)
        public_out = self.public_ffn(modal_input)
        special_out, special_aux_loss = self._construct(self.special_router,self.special_ffn,modal_input,self.specific_expert_num, self.transpose_2dim_ep, self.transpose_4dim_ep)

        output = self.add_3d(special_out,cross_out)
        output = self.add_3d(output,public_out)

        aux_loss = self.add(cross_aux_loss, special_aux_loss)
        return output, aux_loss
