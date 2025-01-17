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
"""load ckpt"""
import numpy as np
from mindspore import transform_checkpoints
from mindspore import context, Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import ops as P
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from scipy import interpolate
from scipy.ndimage.interpolation import zoom


def load_ckpt(args, cfts, net, model, net_with_wrapper, train_dataset, new_epochs,
              valid_dataset=None, is_finetune=False):
    """load ckpt"""
    # load pretrain or resume ckpt
    mode = _get_parallel_mode()
    train_config = args.train_config

    if train_config.resume_ckpt:
        if mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.STAND_ALONE):
            args.logger.info(".........Load Checkpoint..........")
            ckpt_path = cfts.get_checkpoint(train_config.resume_ckpt)
            if is_finetune:
                params_dict = load_pretrained_ckpt(args, net, ckpt_path)
            else:
                params_dict = load_checkpoint(ckpt_path)
            if params_dict.get("epoch_num") and params_dict.get("step_num"):
                args.logger.info(".........Update has trained epochs and steps..........")
                args.train_config.has_trained_epoches = int(params_dict["epoch_num"].data.asnumpy())
                args.train_config.has_trained_steps = int(params_dict["step_num"].data.asnumpy())
            args.logger.info(".........Load Checkpoint To Network..........")
            load_param_into_net(net_with_wrapper, params_dict)

        if mode in (context.ParallelMode.SEMI_AUTO_PARALLEL,):
            args.logger.info(".........Load Model Parallel Checkpoint..........")
            try:
                train_config.resume_ckpt = train_config.resume_ckpt.format(args.local_rank, args.local_rank)
            except Exception as e:
                raise e
            ckpt_path = cfts.get_checkpoint(train_config.resume_ckpt, rank_id=args.local_rank)
            if is_finetune:
                params_dict = load_pretrained_ckpt(args, net, ckpt_path)
            else:
                params_dict = load_checkpoint(
                    ckpt_path, filter_prefix=["adam_v", "adam_m", "epoch_num", "step_num", "global_step"])
            for key in list(params_dict.keys()):
                if 'relative_position_bias' in key:
                    del params_dict[key]
                if 'patch_embed' in key:
                    del params_dict[key]
                if 'decoder' in key:
                    del params_dict[key]

            # for key in list(net.parameters_dict().keys()):
            #     print(key, net.parameters_dict()[key].shape)
            # if is_finetune:
            #     params_dict = load_pretrained_ckpt(args, net, ckpt_path)
            # else:
            #     params_dict = load_torch_ckpt(args, net, ckpt_path)

            if params_dict.get("epoch_num") and params_dict.get("step_num"):
                args.logger.info(".........Update has trained epochs and steps..........")
                args.train_config.has_trained_epoches = int(params_dict["epoch_num"].data.asnumpy())
                args.train_config.has_trained_steps = int(params_dict["step_num"].data.asnumpy())
            args.logger.info(".........Model Build Before Load Model Parallel Checkpoint..........")
            # model.build(train_dataset=train_dataset, valid_dataset=valid_dataset,
            #             epoch=new_epochs, sink_size=train_config.per_epoch_size)
            args.logger.info(".........Load Model Parallel Checkpoint To Network..........")
            not_load = load_param_into_net(net_with_wrapper, params_dict, strict_load=False)
            args.logger.info(".........Not Load Model To Network: {}..........".format(not_load))


def load_torch_ckpt(config, model, torch_ckpt):
    """load pretrained ckpt"""
    config.logger.info(f">>>>>>>>>> torch_ckpt from {torch_ckpt} ..........")
    checkpoint = load_checkpoint(
        torch_ckpt, filter_prefix=["adam_v", "adam_m", "epoch_num", "step_num", "global_step"])
    model_dict = model.parameters_dict()
    # checkpoint_remove_encoder = remove_encoder_keys(checkpoint, logger)
    # model_remove_encoder = remove_encoder_keys(model_dict, logger)
    keys_to_modify = []
    for key in checkpoint.keys():
        if 'mlp.special_ffn.mapping.weight' in key:
            keys_to_modify.append(key)
        elif 'mlp.cross_ffn.mapping.weight' in key:
            keys_to_modify.append(key)
        elif 'mlp.special_ffn.projection.weight' in key:
            keys_to_modify.append(key)
        elif 'mlp.cross_ffn.projection.weight' in key:
            keys_to_modify.append(key)
        elif 'mlp.cross_ffn.projection.bias' in key:
            keys_to_modify.append(key)
        elif 'mlp.special_ffn.projection.bias' in key:
            keys_to_modify.append(key)
        elif 'mlp.cross_ffn.mapping.bias' in key:
            keys_to_modify.append(key)
        elif 'mlp.special_ffn.mapping.bias' in key:
            keys_to_modify.append(key)
        elif 'attn_mask' in key:
            keys_to_modify.append(key)

    for key in keys_to_modify:
        if 'mapping.weight' in key or 'projection.weight' in key:
            v = checkpoint[key]
            v = P.stack((v, v, v, v), 0)
            checkpoint[key] = Parameter(v)
        elif 'bias' in key:
            v = checkpoint[key]
            v = P.expand_dims(v, 0)
            v = P.stack((v, v, v, v), 0)
            v = P.expand_dims(v, 0)
            checkpoint[key] = Parameter(v)
        elif 'attn_mask' in key:
            v = checkpoint[key]
            v = P.expand_dims(v, 0)
            v = P.expand_dims(v, 2)
            checkpoint[key] = Parameter(v)
    config.logger.info(f">>>>>>>>>> loaded successfully '{torch_ckpt}'")
    return checkpoint

def load_pretrained_ckpt(config, model, finetune_ckpt):
    """load pretrained ckpt"""
    config.logger.info(f">>>>>>>>>> Fine-tuned from {finetune_ckpt} ..........")
    checkpoint = load_checkpoint(
        finetune_ckpt, filter_prefix=["adam_v", "adam_m", "epoch_num", "step_num", "global_step"])
    model_dict = model.parameters_dict()
    checkpoint_remove_encoder = remove_encoder_keys(checkpoint, config.logger)
    # model_remove_encoder = remove_encoder_keys(model_dict, logger)

    if config.model.backbone == 'swin' or config.model.backbone == 'swin_v2' :
        config.logger.info(">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model_dict, checkpoint_remove_encoder, config.logger)
    elif config.model.backbone == 'vit':
        config.logger.info(">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_remove_encoder, config.model.depth, config.logger)
    else:
        raise NotImplementedError
    config.logger.info(f">>>>>>>>>> loaded successfully '{finetune_ckpt}'")
    return checkpoint


def remove_encoder_keys(model_dict, logger):
    params_remove_encoder = {}
    if any('encoder.' in k for k in model_dict.keys()):
        params_remove_encoder = {
            k.replace('encoder.', ''): v for k, v in model_dict.items() if k.startswith('encoder.')}
        logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        logger.info('Detect non-pre-trained model, pass without doing anything.')
    return params_remove_encoder


def remap_pretrained_keys_swin(state_dict, checkpoint_model, logger):
    """remap pretrained keys swin"""
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            l1, n_h1 = relative_position_bias_table_pretrained.shape
            l2, n_h2 = relative_position_bias_table_current.shape
            if n_h1 != n_h2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if l1 != l2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(l1 ** 0.5)
                    dst_size = int(l2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0

                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(n_h1):
                        z = relative_position_bias_table_pretrained[:, i].view(
                            src_size, src_size).asnumpy().astype(np.float32)
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(Tensor(f_cubic(dx, dy), mstype.float32).view(-1, 1))

                    new_rel_pos_bias = P.Concat(axis=-1)(tuple(all_rel_pos_bias))
                    new_rel_pos_bias = Parameter(new_rel_pos_bias)
                    checkpoint_model[key] = new_rel_pos_bias

    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def remap_pretrained_keys_vit(model, checkpoint_model, num_layers, logger):
    """remap pretrained keys vit"""
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'encoder.use_rel_pos_bias', False) and \
            "encoder.rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        logger.info("Expand the shared relative position embedding to each transformer block.")
    rel_pos_bias = checkpoint_model["encoder.rel_pos_bias.relative_position_bias_table"]
    rpbt_keys = "encoder.encoder.blocks.%d.attention.relative_position_bias.relative_position_bias_table"
    for i in range(num_layers):
        checkpoint_model[rpbt_keys % i] = rel_pos_bias.clone()
    checkpoint_model.pop("encoder.rel_pos_bias.relative_position_bias_table")

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    model_dict = model.parameters_dict()
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.shape
            dst_num_pos, _ = model_dict[key].shape
            dst_patch_shape = model.encoder.patch_embed.grid_size
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError("Patch must be same.")
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info(
                    "Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                logger.info("Original positions = %s" % str(x))
                logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).asnumpy().astype(np.float32)
                    f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(Tensor(f_cubic(dx, dy), mstype.float32).view(-1, 1))

                rel_pos_bias = P.Concat(axis=-1)(tuple(all_rel_pos_bias))
                new_rel_pos_bias = P.Concat(axis=0)((rel_pos_bias, extra_tokens))
                new_rel_pos_bias = Parameter(new_rel_pos_bias)
                checkpoint_model[key] = new_rel_pos_bias

    return checkpoint_model


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.encoder.patch_embed.num_patches
        num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = P.Reshape()(pos_tokens, (-1, orig_size, orig_size, embedding_size))
            pos_tokens = P.Transpose()(pos_tokens, (0, 3, 1, 2))
            pos_tokens = zoom(pos_tokens.asnumpy(), output=(new_size, new_size), mode='bicubic')
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = P.Transpose()(pos_tokens, (0, 2, 3, 1))
            pos_tokens = pos_tokens.flatten()
            new_pos_embed = P.Concat(axis=1)((extra_tokens, pos_tokens))
            checkpoint_model['pos_embed'] = new_pos_embed
