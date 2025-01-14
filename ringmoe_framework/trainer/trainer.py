# Copyright 2022 Huawei Technologies Co., Ltd
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
"""TrainOneStepWithClipGNAndEMA."""
import logging as logger

from mindspore import nn, Tensor
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from ringmoe_framework.trainer.clip_grad import clip_by_global_norm
from mindspore.parallel._utils import _get_enable_parallel_optimizer
# from ringmoe_framework.trainer.clip_grad_v2 import ClipByGlobalNorm
import mindspore.common.dtype as mstype
from ringmoe_framework.trainer.ema import EMACell
from mindspore.nn.wrap.cell_wrapper import PipelineCell, MicroBatchInterleaved, _VirtualDatasetCell
_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Cast()(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


class TrainOneStepWithClipGNAndEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer,
                 use_clip_grad=False, clip_norm=1.0, optim_type=None,
                 scale_sense=1.0, with_ema=False, ema_decay=0.9999):
        super(TrainOneStepWithClipGNAndEMA, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.with_ema = with_ema
        self.clip_norm = clip_norm
        self.use_clip_grad = use_clip_grad
        self.offload_cpu = True if optim_type is not None and optim_type == "AdamWOP" else False
        # self.clip = ClipByGlobalNorm(self.weights, )
        if self.with_ema:
            self.ema_model = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        global_norm = None
        if not overflow:
            if self.use_clip_grad:
                grads, global_norm = clip_by_global_norm(grads, clip_norm=self.clip_norm)
            if self.offload_cpu and global_norm is not None:
                loss = F.depend(loss, self.optimizer(grads, global_norm))
            else:
                loss = F.depend(loss, self.optimizer(grads))
            # self.optimizer(grads)
            if self.with_ema:
                self.ema_model(self.weights)
        else:
            self.print("==========Overflow Now============")
        return loss

class TrainPipelineWithClipGNAndEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer, config,
                 use_clip_grad=False, clip_norm=1.0, optim_type=None,
                 scale_sense=1.0, with_ema=False, ema_decay=0.9999):
        super(TrainPipelineWithClipGNAndEMA, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.with_ema = with_ema
        self.clip_norm = clip_norm
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.use_clip_grad = use_clip_grad
        self.offload_cpu = True if optim_type is not None and optim_type == "AdamWOP" else False
        self.config = config
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.micro_size = config.parallel_config.micro_batch_num
        self.opt_shard = _get_enable_parallel_optimizer()
        # self.clip = ClipByGlobalNorm(self.weights, )
        if self.with_ema:
            self.ema_model = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, x, mask):
        """construct"""
        weights = self.weights
        loss = self.network(x, mask)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        # grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)

        # grads = self.grad(self.network, weights)(x ,mask, scaling_sens_filled )


        # grads = self.grad(self.network, weights)(x ,mask, self.cast(scaling_sens_filled / self.micro_size,mask.dtype))
        grads = self.grad(self.network, weights)(x ,mask, self.cast(scaling_sens_filled / self.micro_size,mstype.float32))

        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads, accu_grads)

        # grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # # apply grad reducer on grads
        # grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        global_norm = None
        if not overflow:
            if self.use_clip_grad:
                grads, global_norm = clip_by_global_norm(grads, clip_norm=self.clip_norm)
            if self.offload_cpu and global_norm is not None:
                loss = F.depend(loss, self.optimizer(grads, global_norm))
            else:
                loss = F.depend(loss, self.optimizer(grads))
            # self.optimizer(grads)
            if self.with_ema:
                self.ema_model(self.weights)
        else:
            self.print("==========Overflow Now============")
        return loss


def build_wrapper(args, net_with_loss, optimizer, log=logger):
    """get_train_one_step cell"""
    train_wrapper_config = args.train_wrapper

    if train_wrapper_config.use_dynamic_loss_scale:
        log.info("=> Using DynamicLossScaleUpdateCell")
        scale_manager = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=train_wrapper_config.loss_scale, scale_factor=2, scale_window=1000)
    else:
        log.info(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{train_wrapper_config.loss_scale}")
        scale_manager = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=train_wrapper_config.loss_scale)

    if args.parallel_config.pipeline_stage > 1:
        log.info("=> Using TrainPipelineWithClipGNAndEMA")
        # # train_model = nn.PipelineCell(train_model, 4)
        # net_with_loss = PipelineCell(net_with_loss, args.parallel_config.micro_batch_num)

        net_with_loss = PipelineCell(MicroBatchInterleaved(net_with_loss, 1), args.parallel_config.micro_batch_num)
        net_with_loss = _VirtualDatasetCell(net_with_loss)
        net_with_loss = TrainPipelineWithClipGNAndEMA(
            net_with_loss, optimizer, config=args, use_clip_grad=train_wrapper_config.use_clip_grad,
            clip_norm=train_wrapper_config.clip_norm, scale_sense=scale_manager,
            optim_type=args.optimizer.optim_name,
            with_ema=train_wrapper_config.use_ema, ema_decay=train_wrapper_config.ema_decay)
    else:
        net_with_loss = TrainOneStepWithClipGNAndEMA(
            net_with_loss, optimizer, use_clip_grad=train_wrapper_config.use_clip_grad,
            clip_norm=train_wrapper_config.clip_norm, scale_sense=scale_manager,
            optim_type=args.optimizer.optim_name,
            with_ema=train_wrapper_config.use_ema, ema_decay=train_wrapper_config.ema_decay)


    return net_with_loss
