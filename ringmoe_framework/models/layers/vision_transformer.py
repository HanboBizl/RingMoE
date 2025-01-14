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
"""vision transformer"""
import mindspore.common.dtype as mstype
import numpy as np
from mindspore import nn, Tensor
from mindspore.ops.primitive import constexpr
# from mindspore.nn.transformer.op_parallel_config import _check_config
##taoht##
from mindspore.parallel._transformer.op_parallel_config import _check_config
# from mindspore.nn.transformer.moe import default_moe_config, _check_moe_config
##taoht##
from mindspore.parallel._transformer.moe import default_moe_config, MoE, _check_moe_config
##taoht##
# from mindspore.nn.transformer.transformer import default_transformer_config, _get_lambda_func
from mindspore.parallel._transformer.transformer import TransformerOpParallelConfig

from mindspore.ops import operations as P

from .block import Block
from .predictor import PredictorLG
from .layers import Identity

##taoht##
def _get_lambda_func(total_layer=None):
    r"""
    A wrapper function of specifying pipeline stage and gradient aggregation fusion. If the total layer
    is not None, for example, set in the transformer model, the pipeline stage setting function will be
    `(layer_id + 0) // (total_layers / parallel_config.pipeline_stage)` for the encoder and,
    `(layer_id + offset) //
    (total_layers / parallel_config.pipeline_stage)` for the decoder, where `offset` is the layers in the encoder.
    """

    def _set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
        r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs an offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
        """
        # override the layers
        if total_layer:
            layers = total_layer
        # Used for the pipeline's stages setting
        if layers < parallel_config.pipeline_stage:
            raise ValueError(f"layers {layers} must be larger than pipeline stage {parallel_config.pipeline_stage}")

        pp_dis = max(layers // parallel_config.pipeline_stage, 1)
        # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
        pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
        network.pipeline_stage = pp_id

        # Used for optimizer's fusion tag
        dis = max(layers // parallel_config.gradient_aggregation_group, 1)
        network.set_comm_fusion((layer_id + offset) // dis + 1)
        # Used for enabling recomputation of the block
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute:
                network.recompute()
        else:
            if parallel_config.recompute.recompute:
                paralel_op_comm_compute = parallel_config.recompute.parallel_optimizer_comm_recompute
                network.recompute(parallel_optimizer_comm_recompute=paralel_op_comm_compute,
                                  mp_comm_recompute=parallel_config.recompute.mp_comm_recompute,
                                  recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

    return _set_parallel_configure_for_layer

##taoht##
default_transformer_config = TransformerOpParallelConfig()
class VisionTransformer(nn.Cell):
    r"""
        VisionTransformer module with multi-layer stacked of `TransformerLayer`, including multihead self
        attention and feedforward layer.
    """

    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 predictor_layer=False,
                 window_size=None,
                 drop_rate=0.,
                 modal_num=1,
                 attention_dropout_rate=0.,
                 hidden_dropout_rate=0.,
                 hidden_act='gelu',
                 weight_init='XavierUniform',
                 init_values=None,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config,
                 logger=None):
        super(VisionTransformer, self).__init__()
        _check_config(parallel_config)
        # _check_moe_config(moe_config, parallel_config)
        self.logger = logger
        self.logger.info("batch size is {}:".format(batch_size))
        hdr = [x.item() for x in np.linspace(0, hidden_dropout_rate, num_layers)]  # stochastic depth decay rule
        self.batch_size = batch_size
        self.predictor_layer = predictor_layer
        self.modal_num =modal_num
        print('predictor layer', self.predictor_layer)
        self.use_moe = (moe_config.expert_num > 1)
        self.add = P.Add()
        self.aux_loss = Tensor(0.0, mstype.float32)
        self.num_layers = num_layers
        self.blocks = nn.CellList()
        self.predictors = nn.CellList()
        parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        print(parallel_config_args)
        for i in range(num_layers):
            block = Block(
                hidden_size=hidden_size,
                batch_size=batch_size,
                ffn_hidden_size=ffn_hidden_size,
                seq_length=seq_length,
                drop_rate=drop_rate,
                modal_num=modal_num,
                attention_dropout_rate=attention_dropout_rate,
                hidden_dropout_rate=hdr[i],
                init_values=init_values,
                weight_init=weight_init,
                layernorm_compute_type=layernorm_compute_type,
                softmax_compute_type=softmax_compute_type,
                window_size=window_size,
                num_heads=num_heads,
                hidden_act=hidden_act,
                post_layernorm_residual=post_layernorm_residual,
                param_init_type=param_init_type,
                moe_config=moe_config,
                parallel_config=parallel_config_args,
                logger =logger)
            # If the user doesn't pass the fusion function, use the default one
            if not lambda_func:
                lambda_func = _get_lambda_func()

            lambda_func(block, layer_id=i, layers=num_layers,
                        offset=offset, parallel_config=parallel_config)
            self.blocks.append(block)
            if self.predictor_layer:
                predictor = PredictorLG(
                    embed_dim=hidden_size,
                    weight_init=weight_init,
                    layernorm_compute_type=layernorm_compute_type,
                    activation=hidden_act,
                    parallel_config=parallel_config)
                lambda_func(predictor, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)

                self.predictors.append(predictor)

        self.softmax = nn.Softmax(axis=2).to_float(softmax_compute_type)
        self.softmax.softmax.shard(((parallel_config.data_parallel, 1, 1),))
        self.softmax.set_comm_fusion(parallel_config.gradient_aggregation_group)
        self.reshape = P.Reshape()
        self.slice = P.Slice().shard(((parallel_config.data_parallel, 1, 1),))
        self.sum = P.ReduceSum().shard(((parallel_config.data_parallel, 1, 1),))
        self.abs = P.Abs().shard(((parallel_config.data_parallel, 1, 1),))
        self.modal_mask = Tensor(np.zeros((batch_size, seq_length, 1)), mstype.float32)

    def construct(self, hidden_states, attention_mask, init_reset=True, batch_valid_length=None,
                  rel_pos_bias=None, is_mm=False):
        modal_mask = self.modal_mask
        output = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                if self.predictor_layer:
                    scores = self.predictors[i](hidden_states)
                    scores = self.reshape(scores, (self.batch_size, -1, 2))
                    b, n, _ = scores.shape
                    modal_mask = self.softmax(scores)
                    modal_mask = self.slice(modal_mask, (0, 0, 0), (b, n, 1))
                    exchange_token = True
                else:
                    exchange_token = False
                hidden_states, aux_loss = self.blocks[i](
                    hidden_states, attention_mask, init_reset, batch_valid_length,
                    rel_pos_bias, modal_mask, exchange_token)

                accum_loss = self.add(accum_loss, aux_loss)

            if self.predictor_layer:
                modal_mask_loss = self.sum(self.abs(modal_mask))
                output = output + (hidden_states, modal_mask_loss, accum_loss,)
            else:
                output = output + (hidden_states, accum_loss,)
            return output

        for i in range(self.num_layers):
            if self.predictor_layer:
                scores = self.predictors[i](hidden_states)
                scores = self.reshape(scores, (self.batch_size, -1, 2))
                b, n, _ = scores.shape
                modal_mask = self.softmax(scores)
                modal_mask = self.slice(modal_mask, (0, 0, 0), (b, n, 1))
                exchange_token = True
            else:
                exchange_token = False
            hidden_states = self.blocks[i](
                hidden_states, attention_mask, init_reset, batch_valid_length,
                rel_pos_bias, modal_mask, exchange_token)
        if self.predictor_layer:
            modal_mask_loss = self.sum(self.abs(modal_mask))
            output = output + (hidden_states, modal_mask_loss,)
        else:
            output = hidden_states
        return output


@constexpr
def check_predictor(index):
    if index % 2 == 0:
        return True
    return False
