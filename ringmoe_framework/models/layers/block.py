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
"""block of ringmoe_framework"""
import mindspore.common.dtype as mstype
import numpy as np
from mindspore import nn, Parameter, Tensor
# from mindspore.nn.transformer.moe import default_moe_config, _check_moe_config
##taoht##
from mindspore.parallel._transformer.moe import default_moe_config, MoE, _check_moe_config
# from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, _check_config
##taoht##
from mindspore.parallel._transformer.op_parallel_config import default_dpmp_config, _check_config

from mindspore.ops import functional as F
from mindspore.ops import operations as P

from ringmoe_framework.models.core.exchange_token import TokenExchange
from .attention import Attention, WindowAttention, WindowAttentionV2, WindowAttentionV3
from .layers import LayerNorm, Dropout, DropPath, Identity
from .mlp import MLP
from .moe import Moe
from .moe_modal import Moe_modal
from .moe_single import Moe_single
from .utils import _ntuple
import aicc_tools as ac
to_2tuple = _ntuple(2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = x.shape
    x = np.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift_size, shift_axis=(1, 2), parallel_config=default_dpmp_config):
        # pylint: disable=W0613
        super(Roll, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.roll = nn.Roll(to_2tuple(shift_size), shift_axis)
        # self.roll.roll.shard(((dp, 1, 1, 1),),)

    def construct(self, x):
        x = self.roll(x)
        return x


class WindowPartition(nn.Cell):
    """WindowPartitionConstruct Cell"""

    def __init__(self, window_size, parallel_config=None):
        super(WindowPartition, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.reshape = P.Reshape()
        # self.transpose = P.Transpose().shard(((dp, 1, 1, 1),))
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))
        
        self.window_size = window_size

    def construct(self, x):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        b, h, w, c = x.shape
        x = self.reshape(x, (b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (b * h * w // (self.window_size ** 2), self.window_size, self.window_size, c))

        return x


class WindowReverse(nn.Cell):
    """WindowReverseConstruct Cell"""

    def __init__(self, parallel_config=None):
        super(WindowReverse, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((dp, 1, 1, 1, 1, 1),))

    def construct(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        b = windows.shape[0] // (h * w // window_size // window_size)
        x = self.reshape(windows, (b, h // window_size, w // window_size, window_size, window_size, -1))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (b, h, w, -1))
        return x



class SwinTransformerBlock(nn.Cell):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        hidden_act (nn.Cell, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm/_LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 hidden_act='gelu', weight_init='normal', norm_layer=LayerNorm,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(SwinTransformerBlock, self).__init__()
        self.use_moe = (moe_config.expert_num > 1)
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.reshape = P.Reshape()
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer((dim,), eps=1e-6)
        self.norm1.shard(((dp, 1, 1),))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path, ndim=1, parallel_config=parallel_config) \
            if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,), eps=1e-6)
        self.norm2.shard(((dp, 1, 1),))

        self.mlp = MLP(hidden_size=dim,
                       ffn_hidden_size=int(dim * mlp_ratio),
                       dropout_rate=drop,
                       weight_init=weight_init,
                       hidden_act=hidden_act,
                       use_dropout=True,
                       parallel_config=parallel_config)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            in_h, in_w = self.input_resolution
            img_mask = np.zeros((1, in_h, in_w, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            # [64, 49, 49] ==> [1, 64, 1, 49, 49]
            attn_mask = np.expand_dims(attn_mask, axis=1)
            attn_mask = np.expand_dims(attn_mask, axis=0)
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False, name="attention_mask")
            self.roll_pos = Roll(self.shift_size, parallel_config=parallel_config)
            self.roll_neg = Roll(-self.shift_size, parallel_config=parallel_config)
        else:
            self.attn_mask = None

        self.dtype = P.DType()
        self.add_3d = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.window_partition = WindowPartition(self.window_size)
        self.window_reverse = WindowReverse()


    def construct(self, x):
        """construct function"""
        h, w = self.input_resolution
        b, _, c = x.shape
        ori_type = self.dtype(x)
        shortcut = x
        x = self.norm1(x)
        x = self.reshape(x, (b, h, w, c))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        x_windows = self.reshape(x_windows,
                                 (-1, self.window_size * self.window_size, c))  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = self.reshape(attn_windows, (-1, self.window_size, self.window_size, c))
        shifted_x = self.window_reverse(attn_windows, self.window_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x

        x = self.reshape(x, (b, h * w, c))

        # FFN
        x = self.drop_path(x)
        x = self.cast(x, ori_type)
        x = self.add_3d(shortcut, x)
        x_tmp = self.norm2(x)
        x_tmp = self.mlp(x_tmp)
        x_tmp = self.drop_path(x_tmp)
        output = self.add_3d(x, x_tmp)
        output = self.cast(output, ori_type)
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

class SwinTransformerV2Block(nn.Cell):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        hidden_act (nn.Cell, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm/_LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, modal_num=1,moe_tag=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 hidden_act='gelu', weight_init='normal', norm_layer=LayerNorm,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(SwinTransformerV2Block, self).__init__()
        self.use_moe = (moe_config.expert_num > 1)
        self.moe_tag = moe_tag
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.dim = dim
        self.input_resolution = input_resolution
        self.modal_num =modal_num
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.reshape = P.Reshape()
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer((dim,), eps=1e-6)
        self.norm1.shard(((dp, 1, 1),))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path, ndim=1, parallel_config=parallel_config) \
            if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,), eps=1e-6)
        self.norm2.shard(((dp, 1, 1),))
        # self.output = nn.CellList()
        if self.use_moe and self.moe_tag:

            if self.modal_num>1:
                self.mlp = Moe_modal(hidden_size=dim,
                                  dropout_rate=drop,
                                  modal_num = modal_num,
                                  ffn_hidden_size=int(dim * mlp_ratio),
                                  weight_init=weight_init,
                                  hidden_act=hidden_act,
                                  use_dropout=True,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                self.mlp = Moe_single(hidden_size=dim,
                                  dropout_rate=drop,
                                  ffn_hidden_size=int(dim * mlp_ratio),
                                  weight_init=weight_init,
                                  hidden_act=hidden_act,
                                  use_dropout=True,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
        else:
            # Feed Forward Network, FFN
            self.mlp = MLP(hidden_size=dim,
                              dropout_rate=drop,
                              ffn_hidden_size=int(dim * mlp_ratio),
                              weight_init=weight_init,
                              hidden_act=hidden_act,
                              use_dropout=True,
                              parallel_config=parallel_config)


        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            in_h, in_w = self.input_resolution
            img_mask = np.zeros((1, in_h, in_w, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            # [64, 49, 49] ==> [1, 64, 1, 49, 49]
            attn_mask = np.expand_dims(attn_mask, axis=1)
            attn_mask = np.expand_dims(attn_mask, axis=0)
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False, name="attention_mask")
            self.roll_pos = Roll(self.shift_size, parallel_config=parallel_config)
            self.roll_neg = Roll(-self.shift_size, parallel_config=parallel_config)
        else:
            self.attn_mask = None

        self.dtype = P.DType()
        self.add_3d = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.window_partition = WindowPartition(self.window_size)
        self.window_reverse = WindowReverse()


    def construct(self, x):
        """construct function"""
        h, w = self.input_resolution
        b, L, c = x.shape
        assert L == h * w, "input feature has wrong size"
        ori_type = self.dtype(x)
        shortcut = x
        x = self.reshape(x, (b, h, w, c))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        x_windows = self.reshape(x_windows,
                                 (-1, self.window_size * self.window_size, c))  # nW*B, window_size*window_size, C


        # # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        #
        # merge windows
        attn_windows = self.reshape(attn_windows, (-1, self.window_size, self.window_size, c))
        shifted_x = self.window_reverse(attn_windows, self.window_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x

        x = self.reshape(x, (b, h * w, c))

        # FFN
        x = self.cast(x, ori_type)
        x = self.norm1(x)
        x = self.drop_path(x)
        x = self.add_3d(shortcut, x)

        aux_loss = None
        if self.use_moe:
            x_tmp, aux_loss = self.mlp(x)
        else:
            x_tmp = self.mlp(x)
        # x_tmp = self.mlp(x)
        x_tmp = self.norm2(x_tmp)
        x_tmp = self.drop_path(x_tmp)
        output = self.add_3d(x, x_tmp)
        output = self.cast(output, ori_type)

        if self.use_moe:
            return output, aux_loss
        else:
            return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

class Block(nn.Cell):
    """Block of ringmoe_framework"""

    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 drop_rate=0.,
                 modal_num=1,
                 attention_dropout_rate=0.,
                 hidden_dropout_rate=0.,
                 window_size=None,
                 post_layernorm_residual=False,
                 init_values=None,
                 weight_init='XavierUniform',
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config,
                 logger=None):
        super(Block, self).__init__()
        _check_config(parallel_config)
        _check_moe_config(moe_config, parallel_config)
        self.logger = logger
        self.use_moe = (moe_config.expert_num > 1)
        self.modal_num = modal_num
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.layernorm1.shard(((parallel_config.data_parallel, 1),))
        self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.layernorm2.shard(((parallel_config.data_parallel, 1),))
        parallel_config_args = parallel_config.dpmp if self.use_moe else parallel_config
        # print(parallel_config)
        self.attention = Attention(batch_size=batch_size,
                                   src_seq_length=seq_length,
                                   tgt_seq_length=seq_length,
                                   hidden_size=hidden_size,
                                   window_size=window_size,
                                   num_heads=num_heads,
                                   weight_init=weight_init,
                                   hidden_dropout_rate=hidden_dropout_rate,
                                   attention_dropout_rate=attention_dropout_rate,
                                   softmax_compute_type=softmax_compute_type,
                                   param_init_type=param_init_type,
                                   parallel_config=parallel_config_args)
        # self.output = nn.CellList()
        if self.use_moe:
            if self.modal_num>1:
                self.output = Moe_modal(hidden_size=hidden_size,
                                  dropout_rate=drop_rate,
                                  modal_num = modal_num,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  weight_init=weight_init,
                                  hidden_act=hidden_act,
                                  use_dropout=False,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                self.output = Moe(hidden_size=hidden_size,
                                  dropout_rate=drop_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  weight_init=weight_init,
                                  hidden_act=hidden_act,
                                  use_dropout=False,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
        else:
            # Feed Forward Network, FFN
            self.output = MLP(hidden_size=hidden_size,
                              dropout_rate=drop_rate,
                              ffn_hidden_size=ffn_hidden_size,
                              param_init_type=param_init_type,
                              weight_init=weight_init,
                              hidden_act=hidden_act,
                              use_dropout=False,
                              parallel_config=parallel_config)
        self.post_layernorm_residual = post_layernorm_residual
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
        self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.dtype = mstype.float16

        if init_values is not None:
            self.gamma_1 = Parameter(
                Tensor(init_values * np.ones((hidden_size,)), mstype.float32),
                name="gamma1", requires_grad=True)
            self.gamma_2 = Parameter(
                Tensor(init_values * np.ones((hidden_size,)), mstype.float32),
                name="gamma2", requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.mul_gamma = P.Mul().shard(((parallel_config.data_parallel, 1), (1,)))
        self.drop_path = Dropout(1 - hidden_dropout_rate)
        self.drop_path.shard(((parallel_config.data_parallel, 1),))
        self.drop_path3d = Dropout(1 - hidden_dropout_rate)
        self.drop_path3d.shard(((parallel_config.data_parallel, 1, 1),))
        self.mul = P.Mul().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.reshape = P.Reshape()
        self.token_exchange = TokenExchange(
            batch_size, seq_length, hidden_size,
            modal_mask_threshold=0.02,
            parallel_config=parallel_config)
        self.add_0d = P.Add().shard(((), ()))

        self.key_past = None
        self.value_past = None

    def construct(self, x, input_mask, init_reset=True, batch_valid_length=None,
                  rel_pos_bias=None, modal_mask=None, exchange_token=False):
        """construct of Block"""
        # self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        attention = self.attention(
            input_x, input_x, input_x, input_mask,
            self.key_past, self.value_past,
            batch_valid_length, rel_pos_bias)

        if exchange_token:
            ori_shape = F.shape(attention)
            attention = self.reshape(attention, (self.batch_size, self.seq_length, -1))
            attention = self.mul(attention, modal_mask)
            attention = self.token_exchange(attention, modal_mask)
            attention = self.reshape(attention, ori_shape)

        if self.gamma_1 is not None:
            attention = self.mul_gamma(attention, self.gamma_1)

        attention = self.drop_path(attention)

        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        # print('output',output_x.shape)
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        if self.gamma_2 is not None:
            mlp_logit = self.mul_gamma(mlp_logit, self.gamma_2)

        mlp_logit = self.drop_path(mlp_logit)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            # mlp_logit = self.drop_path3d(mlp_logit)
            x = P.Reshape()(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            # mlp_logit = self.drop_path(mlp_logit)
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)

        if self.use_moe:
            return output, aux_loss
        return output


