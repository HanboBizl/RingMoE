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
"""build arch"""
from .mae import build_mae
from .ringmo import build_ringmo
from .simmim import build_simmim
from .ringmo_mm import build_ringmo_mm
from .simmim_moe import build_simmim_moe
from .simmim_single_moe import build_simmim_single_moe


def build_model(config):
    if config.arch == 'mae':
        model = build_mae(config)
    elif config.arch == 'simmim':
        model = build_simmim(config)
    elif config.arch == "ringmo_framework":
        model = build_ringmo(config)
    elif config.arch == "ringmo_mm":
        model = build_ringmo_mm(config)
    elif config.arch == "simmim_moe":
        model = build_simmim_moe(config)
    elif config.arch == "simmim_single_moe":
        model = build_simmim_single_moe(config)
    else:
        raise NotImplementedError("This arch {} should not support".format(config.arch))
    return model
