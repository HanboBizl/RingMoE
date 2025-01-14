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
"""relative pos bias of ringmoe_framework"""
import mindspore.common.dtype as mstype
import mindspore.common.initializer as weight_init
import numpy as np
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from typing import List, Optional, Tuple, Union
from ringmoe_framework.models.layers.layers import Linear, Dropout

class RelativePositionBias(nn.Cell):
    """relative position bias"""

    def __init__(self, window_size, num_heads):
        super(RelativePositionBias, self).__init__()

        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

        self.relative_position_bias_table = Parameter(
            weight_init.initializer(
                weight_init.TruncatedNormal(sigma=.02),
                (self.num_relative_distance, num_heads)),
            name='relative_position_bias_table')

        # get pair-wise relative position index for each token inside the window
        coords_h = Tensor(np.arange(window_size[0]), mstype.int32)
        coords_w = Tensor(np.arange(window_size[1]), mstype.int32)
        coords = P.Stack(axis=0)(P.Meshgrid(indexing='ij')((coords_h, coords_w)))  # 2, Wh, Ww
        coords_flatten = P.Flatten()(coords)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = P.Transpose()(relative_coords, (1, 2, 0)).asnumpy()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = \
            np.zeros(((window_size[0] * window_size[1] + 1),) * 2, dtype=int)

        relative_position_index[1:, 1:] = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        relative_position_index = Tensor(relative_position_index, mstype.int32)
        relative_position_index = relative_position_index.view(-1)

        self.relative_position_index = Parameter(
            relative_position_index,
            requires_grad=False, name="relative_position_index")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((1, 1, 1),))

        self.gather = P.Gather().shard(((1, 1), (1,)))

    def construct(self):
        relative_position_index = self.relative_position_index  # .view(-1)
        relative_position_bias = self.gather(self.relative_position_bias_table, relative_position_index, 0)
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1] + 1,
             self.window_size[0] * self.window_size[1] + 1, -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        return relative_position_bias


class RelativePositionBiasForSwin(nn.Cell):
    def __init__(self, window_size, num_heads):
        super(RelativePositionBiasForSwin, self).__init__()
        self.window_size = window_size
        # cls to token & token to cls & cls to cls
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)

        self.relative_position_bias_table = Parameter(
            weight_init.initializer(
                weight_init.TruncatedNormal(sigma=.02),
                (self.num_relative_distance, num_heads)),
            name='relative_position_bias_table')

        # get pair-wise relative position index for each token inside the window
        coords_h = Tensor(np.arange(window_size[0]), mstype.int32)
        coords_w = Tensor(np.arange(window_size[1]), mstype.int32)
        coords = P.Stack(axis=0)(P.Meshgrid(indexing='ij')((coords_h, coords_w)))  # 2, Wh, Ww
        coords_flatten = P.Flatten()(coords)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = P.Transpose()(relative_coords, (1, 2, 0)).asnumpy()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = Tensor(np.sum(relative_coords, axis=-1), mstype.int32)  # Wh*Ww, Wh*Ww
        relative_position_index = relative_position_index.view(-1)

        self.relative_position_index = Parameter(
            relative_position_index, requires_grad=False, name="relative_position_index")

        self.reshape = P.Reshape()
        self.transpose = P.Transpose().shard(((1, 1, 1),))
        self.expand_dim = P.ExpandDims().shard(((1, 1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))

    def construct(self):

        relative_position_bias = self.gather(self.relative_position_bias_table, self.relative_position_index, 0)
        relative_position_bias = self.reshape(
            relative_position_bias,
            (self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = self.expand_dim(relative_position_bias, 0)
        return relative_position_bias

class RelativePositionBiasForSwinv2(nn.Cell):
    def __init__(
        self,
        window_size: Tuple[int, int],
        num_heads: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size  # Wh, Ww
        # mlp to generate continuous relative position bias
        self.num_heads = num_heads
        linear1 = Linear(in_channels=2, out_channels=512, has_bias=True, activation="relu" ,param_init_type=mstype.float32,compute_dtype=mstype.float32)
        linear2 = Linear(in_channels=512, out_channels=num_heads, has_bias=False,param_init_type=mstype.float32,compute_dtype=mstype.float32)
        linear1.shard(strategy_matmul=((1, 1), (1, 1)), strategy_bias=((1, 1), (1,)), strategy_activation=((1, 1),))
        linear2.shard(strategy_matmul=((1, 1), (1, 1)))
        self.cpb_mlp  =  nn.SequentialCell([linear1,linear2])

        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=float)
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=float)
        relative_coords_table = np.stack(np.meshgrid(relative_coords_h, relative_coords_w, indexing="ij"), axis=0)
        relative_coords_table = np.transpose(relative_coords_table, (1, 2, 0))
        relative_coords_table = np.expand_dims(relative_coords_table, axis=0)

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
                np.sign(relative_coords_table) * np.log2(np.abs(relative_coords_table) + 1) / np.log2(8)
        )

        self.relative_coords_table = Parameter(
            Tensor(relative_coords_table, mstype.float32), requires_grad=False, name="relative_coords_table"
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(window_size[0])
        coords_w = np.arange(window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"), axis=0)  # 2, Wh, Ww
        coords_flatten = coords.reshape(coords.shape[0], -1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww
        relative_position_index = relative_position_index.reshape(-1)
        self.relative_position_index = Parameter(
            Tensor(relative_position_index, mstype.int32), requires_grad=False, name="relative_position_index"
        )
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.sigmoid = P.Sigmoid().shard(((1, 1, 1),))
        self.transpose = P.Transpose().shard(((1, 1, 1),))
        self.reshape_4d = P.Reshape().shard(((1, 1, 1, 1),))
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
    def construct(self) -> Tensor:
        x = self.cpb_mlp(self.relative_coords_table)
        x = self.reshape_4d(x, (-1,self.num_heads))
        relative_position_bias = self.gather(x, self.relative_position_index, 0)
        relative_position_bias = self.reshape(relative_position_bias,
                                              (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = self.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = 16 * self.sigmoid(relative_position_bias)

        relative_position_bias = self.expand_dims(relative_position_bias, 0)
        return relative_position_bias

if __name__ == '__main__':
    relative_position= RelativePositionBiasForSwin(window_size=(6, 6) , num_heads=4)
    relative_position_bias=relative_position()
    print(relative_position_bias)