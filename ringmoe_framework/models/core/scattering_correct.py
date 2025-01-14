import mindspore.common.initializer as weight_init
from mindspore import nn
from mindspore import ops as P

from ringmoe_framework.models.layers.layers import LayerNorm


class ScatteringCorrection(nn.Cell):
    r"""Scattering Correction For SAR Image"""

    def __init__(self, in_chans=3, feat_chans=3, parallel_config=None):
        super(ScatteringCorrection, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        initial_conv1 = nn.Conv2d(
            in_chans, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        initial_conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        initial_conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))

        initial_relu = nn.ReLU()
        initial_relu.relu.shard(((dp, 1, 1, 1),))

        self.initial = nn.SequentialCell(initial_conv1, initial_relu)

        initial_bn = nn.BatchNorm2d(feat_chans, eps=1e-5)
        initial_bn.bn_train.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))
        initial_bn.bn_infer.shard(((dp, 1, 1, 1), (1,), (1,), (1,), (1,)))

        initial_conv2 = nn.Conv2d(
            feat_chans, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        initial_conv2.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        initial_conv2.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.scattering_norm = nn.SequentialCell(initial_bn, initial_conv2, initial_relu)

        initial_conv3 = nn.Conv2d(
            feat_chans * 2, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        initial_conv3.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        initial_conv3.bias_add.shard(((dp, 1, 1, 1), (1,)))

        initial_sigmod = nn.Sigmoid()
        initial_sigmod.sigmoid.shard(((dp, 1, 1, 1),))
        self.norm_conv_sigmod = nn.SequentialCell(initial_conv3, initial_sigmod)

        initial_conv4 = nn.Conv2d(
            feat_chans * 2, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        initial_conv4.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        initial_conv4.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.init_conv_sigmod = nn.SequentialCell(initial_conv4, initial_sigmod)

        self.attention_conv = nn.Conv2d(
            feat_chans * 2, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        self.attention_conv.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.attention_conv.bias_add.shard(((dp, 1, 1, 1), (1,)))

        attn_conv1 = nn.Conv2d(
            feat_chans * 2, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        attn_conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        attn_conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))

        attn_conv2 = nn.Conv2d(
            feat_chans, feat_chans, kernel_size=1,
            weight_init=weight_init.TruncatedNormal(sigma=0.02),
            has_bias=True, pad_mode='pad')
        attn_conv2.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        attn_conv2.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.attention = nn.SequentialCell(attn_conv1, initial_relu, attn_conv2, initial_sigmod)
        self.cat = P.Concat(axis=1).shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
        self.add = P.Add().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

    def construct(self, x):
        x_init = self.initial(x)
        x_norm = self.scattering_norm(x)
        x_concat = self.cat((x_init, x_norm))
        x_norm_new = self.add(x_norm, self.multi(self.norm_conv_sigmod(x_concat), x_init))
        x_init_new = self.add(x_init, self.multi(self.init_conv_sigmod(x_concat), x_norm))
        x_concat_new = self.cat((x_norm_new, x_init_new))
        x_out = self.multi(self.attention_conv(x_concat_new), self.attention(x_concat_new))
        return x_out
