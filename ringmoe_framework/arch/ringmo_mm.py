import os
import numpy as np

import mindspore.common.initializer as weight_init
from mindspore import dtype as mstype
from mindspore import ops as P
from mindspore import nn, Tensor
from mindspore.common.parameter import Parameter

from .simmim import SimMIM, VisionTransformerForSimMIM
from ringmoe_framework.loss.loss import MSELoss, InfoNceLoss
from ringmoe_framework.models.backbone.swin_transformer import SwinTransformer
from ringmoe_framework.models.core.scattering_correct import ScatteringCorrection
from ringmoe_framework.models.layers.mlp import MLP
from ringmoe_framework.models.layers.patch import Patchify
from ringmoe_framework.models.core.depth2space import DepthToSapce
from ringmoe_framework.models.core.repeat_elements import RepeatElement


class SwinTransformerForRingMo(SwinTransformer):
    def __init__(self, **kwargs):
        super(SwinTransformerForRingMo, self).__init__(**kwargs)

        assert self.num_classes == 0
        dp = self.parallel_config.data_parallel
        # mp = self.parallel_config.model_parallel
        self.mask_token = Parameter(
            weight_init.initializer(weight_init.TruncatedNormal(sigma=.02), (1, 1, self.embed_dim)),
            name='mask_token', requires_grad=True
        )
        self.broadcast = P.BroadcastTo((self.batch_size, self.num_patches, -1)).shard(((1, 1, 1),))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))
        self.reshape = P.Reshape()
        self.sub = P.Sub().shard(((), (dp, 1, 1)))
        self.add = P.Add().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add_pos = P.Add().shard(((dp, 1, 1), (1, 1, 1)))
        self.multi = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.hw = int(self.final_seq ** 0.5)

    def construct(self, x, mask):
        x = self.patch_embed(x)

        mask_tokens = self.broadcast(self.mask_token)
        # self.summary("mask_token", self.mask_token)
        w = self.reshape(mask, (-1,))
        w = self.reshape(w, (x.shape[0], -1))
        w = self.expand_dim(w, -1)
        w = self.cast(w, mstype.float32)

        # w-3dims x-3dims mask_tokens-3dims
        part_xa = self.sub(1, w)  # 1-w
        part_xa = self.multi(x, part_xa)  # x * (1-w)
        part_xb = self.multi(mask_tokens, w)  # mask_tokens * w
        x = self.add(part_xa, part_xb)  # part_xa + part_xb

        if self.ape:
            x = self.add_pos(x, self.absolute_pos_embed)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
        return x

    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForRingMoMM(VisionTransformerForSimMIM):
    def __init__(self, **kwargs):
        super(VisionTransformerForRingMoMM, self).__init__(**kwargs)
        self.modal_mask = Tensor(0, mstype.float32)

    def construct(self, x, mask, is_mm):
        x = self.patch_embed(x)

        batch, seq, channel = x.shape

        mask_tokens = self.broadcast(self.mask_token)  # P.BroadcastTo((B, L, -1))(self.mask_token)
        # self.summary_3d("mask_token", self.mask_token)

        w = self.expand_dim(mask, -1)
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
                x, modal_mask, aux_loss = self.encoder(
                        x, self.encoder_input_mask, rel_pos_bias=rel_pos_bias, is_mm=is_mm)
            else:
                x, modal_mask = self.encoder(
                        x, self.encoder_input_mask, rel_pos_bias=rel_pos_bias, is_mm=is_mm)
        else:
            if self.use_moe:
                x, modal_mask, aux_loss = self.encoder(x, self.encoder_input_mask, is_mm=is_mm)
            else:
                x, modal_mask = self.encoder(x, self.encoder_input_mask, is_mm=is_mm)

        x = self.norm(x)

        x = self.slice(x, (0, 1, 0), (batch, seq, channel))  # x = x[:, 1:]
        x = self.transpose(x, (0, 2, 1))
        x = self.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))

        # x -> [B,C,H,W]
        return x, modal_mask, aux_loss


class RingMoMM(nn.Cell):
    def __init__(self, encoder, encoder_stride, temperature=0.1, out_dim=512, use_contranst=True,
                 modal_num=1, norm_pixel_loss=True, lamda=1e-6, clr_loss_weight=0.2, parallel_config=None):
        super(RingMoMM, self).__init__()
        self.modal_num = modal_num
        self.lamda = lamda
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1

        self.decoder = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        # encoder output -> [B,C,H,W]
        self.decoder.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.decoder.bias_add.shard(((dp, 1, 1, 1), (1,)))

        self.pixelshuffle = DepthToSapce(self.encoder_stride, parallel_config=parallel_config)
        self.pixelshuffle.set_comm_fusion(parallel_config.gradient_aggregation_group)
        self.patchify = Patchify(self.encoder.patch_size, parallel_config=parallel_config)
        self.patchify.set_comm_fusion(parallel_config.gradient_aggregation_group)

        self.mse_loss = MSELoss(parallel_config, norm_pixel_loss)
        self.multi_0d = P.Mul().shard(((), ()))
        self.multi_1d = P.Mul().shard(((), (1,)))
        self.add = P.Add().shard(((), ()))
        self.add_1d = P.Add().shard(((), (1,)))
        self.cat = P.Concat(axis=0).shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.cat2d = P.Concat(axis=0).shard(((1, 1), (1, 1)))
        self.gather = P.GatherD().shard(((dp, 1), (dp, 1)))
        self.sum = P.ReduceSum().shard(((dp, 1, 1),))
        self.abs = P.Abs().shard(((dp, 1, 1),))
        self.scattering = ScatteringCorrection(in_chans=3, feat_chans=3, parallel_config=parallel_config)
        self.scattering.set_comm_fusion(parallel_config.gradient_aggregation_group)
        self.use_contranst = use_contranst
        self.sim_clr = InfoNceLoss(temperature=temperature,
                                   batch_size=encoder.batch_size,
                                   parallel_config=parallel_config.dp_mp_config)
        self.clr_loss_weight = clr_loss_weight
        self.transpose = P.Transpose().shard(((parallel_config.data_parallel, 1, 1, 1),))
        self.mlp = MLP(encoder.embed_dim, out_features=out_dim, hidden_act='relu', parallel_config=parallel_config)

        self.mm_cat = MMConcat(parallel_config.data_parallel)
        self.modal_num_flag = Tensor([1], mstype.int32)
        self.not_equal = P.NotEqual().shard(((1,), (1,)))
        self.slice = P.Slice().shard(((1,),))
        # self.logical_not = P.LogicalNot().shard(((1,),))
        self.is_mm = self.modal_num > 1

    def ringmomm_loss(self, x, x_rec, mask, modal_mask, is_mm):
        x = self.cast(x, mstype.float32)
        x_rec = self.cast(x_rec, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        ringmo_loss = self.mse_loss(x_rec, x, mask)
        if self.lamda > 0 and is_mm:
            # l1_loss = self.sum(self.abs(modal_mask))
            ringmo_loss = self.add(ringmo_loss, self.multi_0d(modal_mask, self.lamda))
        return ringmo_loss

    def _check_input(self, inputs):
        x_input = []
        mask_input = []
        ids_input = []
        for i in range(self.modal_num):
            x_input.append(inputs[i])
            ids_input.append(inputs[-1])
            mask_input.append(inputs[-2])
        is_mm = self.is_mm
        return x_input, mask_input, ids_input, is_mm

    def construct(self, *inputs):  # 数据集： opt sar modal_num == opt_1 opt_2 modal_num
        x_in, mask_in, ids_in, is_mm = self._check_input(inputs)
        ori_in, mask_in, ids_in = self.mm_cat(x_in, mask_in, ids_in)
        mask = self.gather(mask_in, 1, ids_in)
        x_patches = self.patchify(ori_in)
        x = ori_in
        if is_mm:
            x_ = self.scattering(x_in[1])
            x = self.cat((x_in[0], x_))

        # x -> [B,L,C]
        z, modal_mask, moe_loss = self.encoder(x, mask, is_mm=is_mm)
        # z -> [B,C,H,W]
        x_rec = self.decoder(z)
        # self.summary_4d("decoder_conv2d", self.decoder.weight)
        # z -> [B,C,H,W]
        x_rec = self.pixelshuffle(x_rec)

        # patchify imgs
        x_rec_patches = self.patchify(x_rec)

        ringmo_loss = self.ringmomm_loss(x_patches, x_rec_patches, mask, modal_mask, is_mm=is_mm)
        sim_loss = self.add(ringmo_loss, moe_loss)

        # contrastive loss
        if self.use_contranst and is_mm:
            z = self.transpose(z, (0, 2, 3, 1))
            z_features = self.mlp(z)
            con_loss = self.sim_clr(z_features)
            con_loss = self.multi_1d(self.clr_loss_weight, con_loss)
            # sim_loss = self.multi_0d(1 - self.clr_loss_weight, sim_loss)
            sim_loss = self.add_1d(sim_loss, con_loss)
        print(sim_loss)
        return sim_loss

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


class MMConcat(nn.Cell):
    def __init__(self, dp):
        super(MMConcat, self).__init__()  # 0 卡是所有卡总和

        self.cat = P.Concat(axis=0).shard(((1, 1, dp, 1), (1, 1, dp, 1)))
        self.cat2d = P.Concat(axis=0).shard(((1, dp), (1, dp)))
        # self.cat.add_prim_attr("primitive_target", "CPU")
        # self.cat2d.add_prim_attr("primitive_target", "CPU")

    def construct(self, x, y, z):
        ori_in = self.cat(x)
        mask = self.cat2d(y)
        ids_in = self.cat2d(z)
        return ori_in, mask, ids_in


def build_ringmo_mm(config):
    model_type = config.model.backbone
    if model_type == 'swin':
        encoder = SwinTransformerForRingMo(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
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
        encoder = VisionTransformerForRingMoMM(
            parallel_config=config.parallel_config,
            moe_config=config.moe_config,
            batch_size=config.train_config.batch_size * config.device_num
            if config.parallel.parallel_mode == "semi_auto_parallel" and not config.parallel.full_batch
            else config.train_config.batch_size,
            image_size=config.train_config.image_size,
            patch_size=config.model.patch_size,
            in_chans=config.model.in_chans,
            num_classes=0,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            predictor_layer=config.model.predictor_layer,
            drop_rate=config.model.drop_rate,
            drop_path_rate=config.model.drop_path_rate,
            use_abs_pos_emb=config.model.use_abs_pos_emb,
            init_values=config.model.init_values,
            use_rel_pos_bias=config.model.use_rel_pos_bias,
            use_shared_rel_pos_bias=config.model.use_shared_rel_pos_bias)
        encoder_stride = config.model.patch_size
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = RingMoMM(
        encoder=encoder, encoder_stride=encoder_stride, modal_num=config.model.modal_num,
        norm_pixel_loss=config.model.norm_pixel_loss, lamda=config.model.lamda,
        use_contranst=config.use_contranst,
        temperature=config.model.temperature,
        out_dim=config.model.out_dim,
        clr_loss_weight=config.model.clr_loss_weight,
        parallel_config=config.parallel_config)

    return model
