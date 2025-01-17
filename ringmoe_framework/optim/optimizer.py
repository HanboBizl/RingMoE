# Copyright 2022 Huawei Technologies Co., Ltd
# Copyright 2022 Aerospace Information Research Institute,
# Chinese Academy of Sciences.
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
"""optimizer of ringmo"""
import numpy as np
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.common.initializer import initializer
from mindspore import Tensor
from mindspore import Parameter, ParameterTuple
from mindspore.nn import Optimizer
from mindspore.nn import AdamWeightDecay

_adam_opt = ops.MultitypeFuncGraph("adam_opt")
host_assign = ops.Assign()
host_assign.add_prim_attr("primitive_target", "CPU")
host_cast = ops.Cast()
host_cast.add_prim_attr("primitive_target", "CPU")
device_cast = ops.Cast()


@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_kernel(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flags, optim_filter):
    """
    Update parameters by AdamWeightDecay op.
    """
    success = True
    if optim_filter:
        param32 = host_cast(param, mstype.float32)
        gradient = device_cast(gradient, mstype.float32)
        if decay_flags:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
        else:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, 0.0, gradient)
        ret = host_assign(param, host_cast(ops.depend(param32, next_param), ops.dtype(param)))
        return ops.depend(success, ret)
    return success


class AdamWeightDecayOp(Optimizer):
    """adam weight decay op"""

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.clone_param32(prefix="adam_m", init='zeros')
        self.moments2 = self.clone_param32(prefix="adam_v", init='zeros')
        self.opt = ops.AdamWeightDecay()
        self.hyper_map = ops.HyperMap()
        self.opt.add_prim_attr("primitive_target", "CPU")

    def construct(self, gradients):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps),
                                                lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr),
                                                self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr,
                                                        self.weight_decay), self.parameters, self.moments1,
                                            self.moments2,
                                            gradients, self.decay_flags, self.optim_filter)
        return optim_result

    def clone_param32(self, prefix, init=None):
        """clone param32 of AdamWeightDecayOp"""
        new = []
        for old_param in self.parameters:
            param_init = init
            if init is None:
                param_init = old_param.init
            new_state = Parameter(initializer(param_init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.is_param_ps = old_param.is_param_ps
            new_state.init_in_server = old_param.init_in_server
            new_state.cache_enable = old_param.cache_enable
            new_state.requires_aggr = old_param.requires_aggr
            if old_param.cache_shape:
                new_state.cache_shape = old_param.cache_shape
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PanGu-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)
