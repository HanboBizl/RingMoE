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
"""loss functions"""

import os

from mindspore import nn, context, Tensor
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.loss import CrossEntropyLoss
# from mindspore.nn.transformer.loss import CrossEntropyLoss
from mindspore.ops import functional as F


class InfoNceLoss(nn.Cell):
    def __init__(self, temperature=0.1, batch_size=64, n_views=2, parallel_config=None):
        super(InfoNceLoss, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
        else:
            dp = 1
            mp = 1

        self.batch_size = batch_size // 2
        self.temperature = temperature
        self.n_views = n_views
        self.norm = P.L2Normalize(axis=-1).shard(((dp, 1),))
        self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (mp, 1)))
        parallel_config.model_parallel = 1
        self.cross_entropy = CrossEntropyLoss(parallel_config=parallel_config)
        self.reshape = P.Reshape()
        self.gather = P.GatherNd().shard(((1, 1), (1, 1)))
        self.cat = P.Concat(axis=2).shard(((1, 1, 1), (1, 1, 1)))

        self.pos_mask = Tensor(
            [[i, j]
             for i in range(self.batch_size * self.n_views)
             for j in range(self.batch_size * self.n_views)
             if j % self.batch_size == i % self.batch_size and j != i], mstype.int32)
        self.neg_mask = Tensor(
            [[i, j]
             for i in range(self.batch_size * self.n_views)
             for j in range(self.batch_size * self.n_views)
             if j % self.batch_size != i % self.batch_size], mstype.int32)

        # print("pos_mask", self.pos_mask.shape)
        # print("neg_mask", self.neg_mask.shape)

        self.ones_like = P.OnesLike().shard(((dp,),))
        self.zeros = P.Zeros().shard(((dp,),))
        self.real_div = P.RealDiv().shard(((dp, 1), ()))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))

    def construct(self, features):
        b = self.batch_size
        n = self.n_views
        features = self.reshape(features, (b * n, -1))
        # [ B * N, B * N ]
        features = self.norm(features)
        # [ B * N, E ]
        similarity_matrix = self.matmul(features, features)
        # [ B * N, E ] * [ E, B * N ] = [ B * N, B * N ]

        pos = self.gather(similarity_matrix, self.pos_mask)
        # [ B * N, N - 1 ]
        neg = self.gather(similarity_matrix, self.neg_mask)
        # [ B * N, (B - 1) * N ]

        pos = self.reshape(pos, (b * n, -1))
        neg = self.reshape(neg, (b * n, -1))
        pos = self.expand_dim(pos, 0)
        neg = self.expand_dim(neg, 0)
        logits = self.cat((pos, neg))
        logits = self.reshape(logits, (logits.shape[1], -1))

        labels = self.zeros(logits.shape[0], mstype.int32)
        logits = self.real_div(logits, self.temperature)
        input_mask = self.ones_like(labels)
        input_mask = self.cast(input_mask, mstype.float32)
        return self.cross_entropy(logits, labels, input_mask)


class L1Loss(nn.Cell):
    def __init__(self, reduction='mean', parallel_config=None):
        super(L1Loss, self).__init__()

        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1

        self.abs = P.Abs().shard(((dp, 1, 1, 1),))
        self.sub = P.Sub().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.mul = P.Mul().shard(((), (dp, 1, 1, 1)))
        self.reduce_mean = P.ReduceMean().shard(((dp, 1, 1, 1),))
        self.reduce_sum = P.ReduceSum().shard(((dp, 1, 1, 1),))
        self.cast = P.Cast()

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, logits, labels):
        x_sub = self.sub(logits, labels)
        x = self.abs(x_sub)
        return self.get_loss(x)

class MSELoss(nn.Cell):
    def __init__(self, parallel_config, norm_pixel_loss=False):
        super(MSELoss, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.add_loss = P.Add().shard(((dp, 1, 1), ()))
        self.sub = P.Sub().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.divide = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
        self.pow = P.Pow().shard(((dp, 1, 1), ()))
        self.divide1 = P.RealDiv().shard(((), ()))
        self.divide2 = P.RealDiv().shard(((dp, 1, 1), ()))
        self.square = P.Square().shard(((dp, 1, 1, 1 ),))
        self.cast = P.Cast()
        self.mean1 = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.mean2 = P.ReduceMean().shard(((dp, 1, 1),))
        self.mul = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.sum = P.ReduceSum().shard(((dp, 1, 1, 1),))
        self.sum2 = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1),))
        self.norm_pixel_loss = norm_pixel_loss
        self.reshape = P.Reshape().shard(((dp, 1, 1, 1),))
    def construct(self, pred, target, mask):
        # pred = self.reshape(pred, (target.shape[0], target.shape[1], target.shape[2]))
        # mask = self.reshape(mask, (target.shape[0], target.shape[1], target.shape[2]))
        pred = self.cast(pred, mstype.float32)
        target = self.cast(target, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        if self.norm_pixel_loss:
            mean = self.mean1(target, -1)
            var = self.variance(target)
            # var = target.var(keepdims=True, axis=-1)
            var = self.add_loss(var, 1e-6)
            std = self.pow(var, 0.5)
            sub = self.sub(target, mean)
            target = self.divide(sub, std)
        res = self.sub(pred, target)
        recon_loss = self.square(res)
        # recon_loss = self.mean2(recon_loss, -1)
        loss_mask = self.mul(recon_loss, mask)
        loss_sum = self.sum(loss_mask)
        mask_sum = self.sum(mask)
        loss = self.divide1(loss_sum, mask_sum)
        return loss

    def variance(self, x):
        axis = (x.ndim - 1,)
        x_mean = self.mean1(x, axis)
        x_sub = self.sub(x, x_mean)
        x_pow = self.pow(x_sub, 2)
        x_sum = self.sum2(x_pow, axis)
        x_var = self.divide2(x_sum, x.shape[-1])
        return x_var


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000, aux_factor=0.4):
        super().__init__()
        self.aux_factor = aux_factor
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.shape = P.Shape()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logits, label):
        if isinstance(logits, tuple):
            logit, aux_logit = logits
        else:
            logit, aux_logit = logits, None

        if self.sparse:
            label = self.onehot(label, self.shape(logit)[1], self.on_value, self.off_value)

        loss = self.ce(logit, label)
        if aux_logit is not None:
            loss = loss + self.aux_factor * self.ce(aux_logit, label)
        return loss


class CrossEntropySmoothMixup(LossBase):
    """CrossEntropy"""

    def __init__(self, reduction='mean', smooth_factor=0., num_classes=1000):
        super().__init__()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = 1.0 * smooth_factor / (num_classes - 2)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        off_label = P.Select()(P.Equal()(label, 0.0), \
                               P.Fill()(mstype.float32, P.Shape()(label), self.off_value), \
                               P.Fill()(mstype.float32, P.Shape()(label), 0.0))

        label = self.on_value * label + off_label
        loss = self.cross_entropy(logit, label)
        return loss


class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment"""

    def __init__(self, parallel_config=None):
        super(SoftTargetCrossEntropy, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.mean_ops = P.ReduceMean(keep_dims=False).shard(((1,),))
        self.sum_ops = P.ReduceSum(keep_dims=False).shard(((dp, 1),))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul1d = P.Mul().shard(((dp, 1), ()))
        self.log_softmax = P.LogSoftmax().shard(((dp, 1),))

    def construct(self, logit, label):
        logit = P.Cast()(logit, mstype.float32)
        label = P.Cast()(label, mstype.float32)
        logit_softmax = self.log_softmax(logit)
        neg_target = self.mul1d(label, -1)
        soft_target = self.mul(neg_target, logit_softmax)
        loss = self.sum_ops(soft_target, -1)
        return self.mean_ops(loss)


class CrossEntropyIgnore(LossBase):
    """CrossEntropyIgnore"""

    def __init__(self, num_classes=21, ignore_label=255):
        super().__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


def get_loss(args):
    """get_loss"""
    loss = None
    if args.loss_type == 'ce_smooth':
        loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                                  num_classes=args.num_classes,
                                  aux_factor=args.aux_factor)
    elif args.loss_type == 'ce_smooth_mixup':
        loss = CrossEntropySmoothMixup(smooth_factor=args.label_smooth_factor,
                                       num_classes=args.num_classes)
    elif args.loss_type == 'ce_ignore':
        loss = CrossEntropyIgnore(num_classes=args.num_classes,
                                  ignore_label=args.ignore_label)
    elif args.loss_type == 'soft_ce':
        loss = SoftTargetCrossEntropy(parallel_config=args.parallel_config)
    else:
        raise NotImplementedError

    return loss
