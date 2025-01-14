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
"""simmim of ringmoe_framework"""
import mindspore.common.initializer as weight_init
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindspore.common.parameter import Parameter

from ringmoe_framework.loss.loss import L1Loss
from ringmoe_framework.models.backbone.swin_transformer import SwinTransformer
from ringmoe_framework.models.backbone.swin_transformerv2 import SwinTransformerV2
from ringmoe_framework.models.backbone.vit import Vit
from ringmoe_framework.models.core.depth2space import DepthToSapce
from ringmoe_framework.models.core.repeat_elements import RepeatElement

import aicc_tools as ac

class SwinTransformerForSimMIM_Moe(SwinTransformer):
    """swin transformer for simmim"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        dp = self.parallel_config.data_parallel
        self.mask_token = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, 1, self.embed_dim)),
            name='mask_token', requires_grad=True)
        self.broadcast = P.BroadcastTo((self.batch_size, self.num_patches, -1)).shard(((1, 1, 1),))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.reshape = P.Reshape()
        self.sub_2 = P.Sub().shard(((), (dp, 1, 1)))
        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.hw = int(self.final_seq ** 0.5)

    def construct(self, x, mask):
        # pylint: disable=W0221
        x = self.patch_embed(x)

        mask_tokens = self.broadcast(self.mask_token)
        # w = self.expand_dim(self.reshape(mask.flatten(), (B, -1)), -1).astype(mstype.float32)
        # x = x * (1. - w) + mask_tokens * w
        # self.summary("mask_token", self.mask_token)
        w = self.reshape(mask, (-1,))
        w = self.reshape(w, (x.shape[0], -1))
        w = self.expand_dim(w, -1)
        w = self.cast(w, mstype.float32)

        # w-3dims x-3dims mask_tokens-3dims
        part_xa = self.sub_2(1, w)  # 1-w
        part_xa = self.multi(x, part_xa)  # x * (1-w)
        part_xb = self.multi(mask_tokens, w)  # mask_tokens * w
        x = self.add(part_xa, part_xb)  # part_xa + part_xb

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        aux_loss = 0.
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.transpose(x, (0, 2, 1))

        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
        return x, aux_loss

    def no_weight_decay(self):  
        return super().no_weight_decay() | {'mask_token'}

class SwinTransformerV2ForSimMIM_Moe(SwinTransformerV2):
    """swin transformer for simmim"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        dp = self.parallel_config.data_parallel

        self.mask_token = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, 1, self.embed_dim)),
            name='mask_token', requires_grad=True)
        self.broadcast = P.BroadcastTo((self.batch_size, self.num_patches, -1)).shard(((1, 1, 1),))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.reshape = P.Reshape()
        self.sub_2 = P.Sub().shard(((), (dp, 1, 1)))
        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.hw = int(self.final_seq ** 0.5)
        self.add_loss = P.Add()
        self.mmcat_4N=MMConcat_4N(dp)

    def construct(self, x_input, mask_input):
        # pylint: disable=W0221

        x0= (self.patch_embed0(x_input[0]))
        x1= (self.patch_embed1(x_input[1]))
        x2= (self.patch_embed2(x_input[2]))
        x3= (self.patch_embed3(x_input[3]))
        x, mask = self.mmcat_4N([x0, x1, x2, x3], mask_input)

        mask_tokens = self.broadcast(self.mask_token)

        w = self.reshape(mask, (-1,))
        w = self.reshape(w, (x.shape[0], -1))
        w = self.expand_dim(w, -1)
        w = self.cast(w, mstype.float32)

        # w-3dims x-3dims mask_tokens-3dims
        part_xa = self.sub_2(1, w)  # 1-w
        part_xa = self.multi(x, part_xa)  # x * (1-w)
        part_xb = self.multi(mask_tokens, w)  # mask_tokens * w
        x = self.add(part_xa, part_xb)  # part_xa + part_xb

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        aux_loss = 0.
        if self.use_moe:
            for layer in self.layers:
                x, aux_loss_i = layer(x)
                aux_loss =self.add_loss(aux_loss, aux_loss_i)
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.norm(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
        return x, aux_loss

    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class VisionTransformerForSimMIM_Moe(Vit):
    """vision transformer for simmim"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0

        dp = self.encoder_config["parallel_config"].data_parallel
        self.mask_token = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, 1, self.embed_dim)),
            name='mask_token', requires_grad=True)
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.reshape = P.Reshape()
        self.cat = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.hw = int(self.num_patches ** 0.5)

        self.broadcast = P.BroadcastTo((self.batch_size, self.seq_length - 1, -1)).shard(((1, 1, 1),))

        # self.sub = P.Sub().shard(((1, 1, 1), (1, 1, 1)))
        self.sub_2 = P.Sub().shard(((), (dp, 1, 1)))
        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))

        self.slice = P.Slice().shard(((dp, 1, 1),))

        self.summary_3d = P.HistogramSummary().shard(((dp, 1, 1),))

        self.logger.info("data_parallel is: {}".format(dp))
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

    def construct(self, x, mask):
        x = self.patch_embed(x)

        batch, seq, channel = x.shape
        # [b,196,1408]

        mask_tokens = self.broadcast(self.mask_token)  # P.BroadcastTo((B, L, -1))(self.mask_token)
        # self.summary_3d("mask_token", self.mask_token)
        w = self.reshape(mask, (-1,))
        w = self.reshape(w, (x.shape[0], -1))
        w = self.expand_dim(w, -1)
        w = self.cast(w, mstype.float32)

        # w-3dims x-3dims mask_tokens-3dims
        part_xa = self.sub_2(1, w)  # 1-w
        part_xa = self.multi(x, part_xa)  # x * (1-w)
        part_xb = self.multi(mask_tokens, w)  # mask_tokens * w
        x = self.add(part_xa, part_xb)  # part_xa + part_xb

        cls_tokens = self.tile(self.cls_tokens, (batch, 1, 1))
        x = self.cat((cls_tokens, x))
        # self.summary_3d("cls_tokens", self.cls_tokens)
        if self.pos_embed is not None:
            x = self.add(x, self.pos_embed)

        x = self.dropout(x)
        aux_loss = 0.
        if self.rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias()
            if self.use_moe:
                x, aux_loss = self.encoder(x, self.encoder_input_mask,  rel_pos_bias=rel_pos_bias)
            else:
                x = self.encoder(x, self.encoder_input_mask, rel_pos_bias=rel_pos_bias)
        else:
            if self.use_moe:
                x, aux_loss = self.encoder(x, self.encoder_input_mask)
            else:
                x = self.encoder(x, self.encoder_input_mask)
        x = self.norm(x)

        x = self.slice(x, (0, 1, 0), (batch, seq, channel))  # x = x[:, 1:]
        x = self.transpose(x, (0, 2, 1))
        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))

        # x -> [B,C,H,W]
        return x, aux_loss


class SimMIM_Moe(nn.Cell):
    """SimMIM"""

    def __init__(self, encoder, encoder_stride, parallel_config=None, modal_num = 1):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.modal_num = modal_num
        self.decoder_0 = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder_0.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder_0.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.decoder_1 = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 8,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder_1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder_1.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.decoder_2 = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder_2.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder_2.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.decoder_3 = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 4,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder_3.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder_3.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.pixelshuffle = DepthToSapce(self.encoder_stride, parallel_config=parallel_config)
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.l1_loss = L1Loss(reduction='none', parallel_config=parallel_config)


        # self.mmcat_3N = MMConcat_3N(dp)
        self.mmcat_4N = MMConcat_4N(dp)
        self.mmsplit_4N = MMSplit_4N(dp)
        self.cat_4d = P.Concat(axis=0).shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.expand_dim = P.ExpandDims().shard(((dp, 1, 1),))
        self.cast = P.Cast()
        self.div = P.Div().shard(((), ()))
        self.multi = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.sum = P.ReduceSum().shard(((dp, 1, 1, 1),))
        self.add = P.Add().shard(((), ()))

        self.repeat_elements_0 = RepeatElement(self.patch_size, 1, parallel_config=parallel_config)
        self.repeat_elements_1 = RepeatElement(self.patch_size, 2, parallel_config=parallel_config)

        self.summary_4d = P.HistogramSummary().shard(((dp, 1, 1, 1),))



    def sim_loss(self, x, x_rec, mask, in_chans):
        """sim loss"""
        x = self.cast(x, mstype.float32)
        x_rec = self.cast(x_rec, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        loss_recon = self.l1_loss(x, x_rec)
        mul_a = self.multi(loss_recon, mask)
        div_a = self.sum(mul_a)
        sum_b = self.sum(mask)
        div_b = self.add(sum_b, 1e-5)
        loss = self.div(div_a, div_b)
        loss = self.div(loss, in_chans)
        return loss

    def compute_sim(self,x, z, mask, in_chans):

        x_rec = self.pixelshuffle(z)

        # mask -3dim
        rp_el_0 = self.repeat_elements_0(mask)
        rp_el_1 = self.repeat_elements_1(rp_el_0)
        mask = self.expand_dim(rp_el_1, 1)

        sim_loss = self.sim_loss(x, x_rec, mask, in_chans)
        return sim_loss
    def _check_input(self, inputs):
        x_input = []
        mask_input = []
        for i in range(self.modal_num):
            x_input.append(inputs[i*2])
            mask_input.append(inputs[i*2+1])
        return x_input, mask_input

    def construct(self, *inputs):
        """construct of SimMIM"""
        # x -> [B,L,C]
        x_input, mask_input = self._check_input(inputs)

        z, moe_loss = self.encoder(x_input, mask_input)

        z0, z1, z2, z3 = self.mmsplit_4N(z)
        # z -> [B,C,H,W]

        z0 = self.decoder_0(z0)
        z1 = self.decoder_1(z1)
        z2 = self.decoder_2(z2)
        z3 = self.decoder_3(z3)
        sim_loss0 = self.compute_sim(x_input[0], z0, mask_input[0], 3)
        sim_loss1 = self.compute_sim(x_input[1], z1, mask_input[1], 8)
        sim_loss2 = self.compute_sim(x_input[2], z2, mask_input[2], 3)
        sim_loss3 = self.compute_sim(x_input[3], z3, mask_input[3], 4)

        sim_loss = self.add(sim_loss0, sim_loss1)
        sim_loss = self.add(sim_loss, sim_loss2)
        sim_loss = self.add(sim_loss, sim_loss3)
        sim_loss = self.add(sim_loss, moe_loss)

        return sim_loss

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

class MMConcat_3N(nn.Cell):
    def __init__(self, dp):
        super(MMConcat_3N, self).__init__()  # 0 卡是所有卡总和

        self.cat = P.Concat(axis=0).shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.cat2d = P.Concat(axis=0).shard(((1, 1, 1), (1, 1, 1)))
        # self.cat.add_prim_attr("primitive_target", "CPU")
        # self.cat2d.add_prim_attr("primitive_target", "CPU")

    def construct(self, x, y):
        ori_in = self.cat((x[0], x[1]))
        ori_in = self.cat((ori_in, x[2]))
        mask = self.cat2d((y[0], y[1]))
        mask = self.cat2d((mask, y[2]))
        return ori_in, mask

class MMConcat_4N(nn.Cell):
    def __init__(self, dp):
        super(MMConcat_4N, self).__init__()  # 0 卡是所有卡总和

        # self.cat = P.Concat(axis=0).shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.cat2d = P.Concat(axis=0).shard(((1, 1, 1), (1, 1, 1)))
        # self.cat.add_prim_attr("primitive_target", "CPU")
        # self.cat2d.add_prim_attr("primitive_target", "CPU")

    def construct(self, x, y):
        ori_in = self.cat2d((x[0], x[1]))
        ori_in = self.cat2d((ori_in, x[2]))
        ori_in = self.cat((ori_in, x[3]))
        mask = self.cat2d((y[0], y[1]))
        mask = self.cat2d((mask, y[2]))
        mask = self.cat2d((mask, y[3]))
        return ori_in, mask

class MMSplit_4N(nn.Cell):
    def __init__(self, dp):
        super(MMSplit_4N, self).__init__()  # 0 卡是所有卡总和

        self.split = P.Split(axis=0, output_num=3).shard(((1, 1, 1, 1),))


    def construct(self, x):
        output = self.split(x)
        # return output[0], output[1], output[2]
        return output[0], output[1], output[2], output[3]

def simmim_vit_base_p16(**kwargs):
    encoder = VisionTransformerForSimMIM_Moe(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return SimMIM_Moe(encoder=encoder, encoder_stride=16)


def simmim_vit_large_p16(**kwargs):
    encoder = VisionTransformerForSimMIM_Moe(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return SimMIM_Moe(encoder=encoder, encoder_stride=16)


def simmim_swin_tiny_p4_w6(**kwargs):
    encoder = SwinTransformerForSimMIM_Moe(
        image_size=192, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=6, mlp_ratio=4, **kwargs)
    return SimMIM_Moe(encoder=encoder, encoder_stride=32)


def simmim_swin_tiny_p4_w7(**kwargs):
    encoder = SwinTransformerForSimMIM_Moe(
        image_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=6, mlp_ratio=4, **kwargs)
    return SimMIM_Moe(encoder=encoder, encoder_stride=32)


def simmim_swin_base_p4_w6(**kwargs):
    encoder = SwinTransformerForSimMIM_Moe(
        image_size=192, patch_size=4, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], window_size=6, mlp_ratio=4, **kwargs)
    return SimMIM_Moe(encoder=encoder, encoder_stride=32)


def simmim_swin_base_p4_w7(**kwargs):
    encoder = SwinTransformerForSimMIM_Moe(
        image_size=224, patch_size=4, embed_dim=128, depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4, **kwargs)
    return SimMIM_Moe(encoder=encoder, encoder_stride=32)


def build_simmim_moe(config):
    """build simmim"""
    model_type = config.model.backbone
    logger = ac.get_logger()
    logger.info(model_type)
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM_Moe(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            modal_num=config.model.modal_num,
            batch_size=config.train_config.batch_size * config.device_num*config.model.modal_num
            if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size*config.model.modal_num,
            image_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
            embed_dim=config.model.embed_dim,
            depths=config.model.depth,
            num_heads=config.model.num_heads,
            window_size=config.model.window_size,
            mlp_ratio=config.model.mlp_ratio,
            qkv_bias=config.model.qkv_bias,
            qk_scale=config.model.qk_scale,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            ape=config.model.ape,
            patch_norm=config.model.patch_norm)
        encoder_stride = 32
    elif  model_type == 'swin_v2':
        encoder = SwinTransformerV2ForSimMIM_Moe(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            modal_num=config.model.modal_num,
            batch_size=config.train_config.batch_size * config.parallel_config.data_parallel * config.model.modal_num
            if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size * config.model.modal_num ,
            image_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
            embed_dim=config.model.embed_dim,
            depths=config.model.depth,
            num_heads=config.model.num_heads,
            window_size=config.model.window_size,
            mlp_ratio=config.model.mlp_ratio,
            qkv_bias=config.model.qkv_bias,
            qk_scale=config.model.qk_scale,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            ape=config.model.ape,
            patch_norm=config.model.patch_norm)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM_Moe(
            logger =logger,
            modal_num = config.model.modal_num,
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            batch_size=config.train_config.batch_size * config.device_num*config.model.modal_num
            if config.parallel.parallel_mode == "semi_auto_parallel" else config.train_config.batch_size*config.model.modal_num,
            image_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            use_abs_pos_emb=config.model.use_abs_pos_emb,
            init_values=config.model.init_values,
            use_rel_pos_bias=config.model.use_rel_pos_bias,
            use_shared_rel_pos_bias=config.model.use_shared_rel_pos_bias)
        encoder_stride = config.model.patch_size
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM_Moe(encoder=encoder, encoder_stride=encoder_stride, parallel_config=config.parallel_config, modal_num = config.model.modal_num)

    return model
