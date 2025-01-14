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
"""pretrain of ringmoe_framework"""
import argparse
import os

import aicc_tools as ac
from mindspore.train.model import Model
# from mindspore import nn
from register import RingMoEConfig, ActionDict
from ringmoe_framework.arch import build_model
from ringmoe_framework.datasets import build_dataset
from ringmoe_framework.lr import build_lr
from ringmoe_framework.monitors.callback import build_pretrain_callback
from ringmoe_framework.optim import build_optim
from ringmoe_framework.parallel_config import build_parallel_config
from ringmoe_framework.tools.helper import build_context, str2bool
from ringmoe_framework.tools.helper import count_params
from ringmoe_framework.tools.load_ckpt import load_ckpt
from ringmoe_framework.trainer import build_wrapper

ac.obs_register(ak='RWQOHBTYFU8IC4QR5BXX',
                sk='dDzLnOg7fyXkqs9cApYPDO8ZmrRhL8ZLGazGrCF8',
                server='https://obs.cn-south-222.ai.pcl.cn')


# @ac.aicc_monitor
def main(args):
    # init context
    cfts, profile_cb = build_context(args)

    # build dataset
    args.logger.info(".........Build Dataset..........")
    dataset = build_dataset(args)
    step_per_epoch = dataset.get_dataset_size()
    new_epochs = 1

    actual_epoch_num = int(new_epochs * step_per_epoch / args.train_config.callback_step)

    args.data_size = step_per_epoch
    args.logger.info("Actual_epoch_num epochs:{}".format(actual_epoch_num))
    args.logger.info("Create training dataset finish, data size:{}".format(step_per_epoch))

    # build context config
    args.logger.info(".........Build context config..........")
    build_parallel_config(args)
    args.logger.info("context config is:{}".format(args.parallel_config))
    args.logger.info("moe config is:{}".format(args.moe_config))

    # build net
    args.logger.info(".........Build Net..........")
    net = build_model(args)
    args.logger.info("网络参数量：{} M.".format(count_params(net)))



    # build lr
    args.logger.info(".........Build LR Schedule..........")
    lr_schedule = build_lr(args)

    # define optimizer
    args.logger.info(".........Build Optimizer..........")
    optimizer = build_optim(args, net, lr_schedule, args.logger)

    # define model
    args.logger.info(".........Build Train Model..........")
    train_model = build_wrapper(args, net, optimizer, log=args.logger)

    # define Model and begin training
    args.logger.info(".........Starting Init Train Model..........")
    model = Model(train_model)

    # resume ckpt
    load_ckpt(args, cfts, net, model, train_model, dataset, actual_epoch_num)

    # define callback
    callback = build_pretrain_callback(args, cfts)

    if args.profile:
        callback.append(profile_cb)

    args.logger.info(".........Starting Training Model..........")
    model.train(actual_epoch_num, dataset, callbacks=callback,
                dataset_sink_mode=args.train_config.sink_mode,
                sink_size=args.train_config.callback_step) 



if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(
            work_path, "config/simmim_pcl/pretrain_simmim_swinv2_giant_p4_w12_aircas_192_200ep_moe_mm.yaml"),
        help='YAML config files')
    parser.add_argument('--device_id', default=None, type=int, help='device id')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--use_parallel', default=None, type=str2bool, help='whether use parallel mode')
    parser.add_argument('--profile', default=None, type=str2bool, help='whether use profile analysis')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    config = RingMoEConfig(args_.config)
    if args_.device_id is not None:
        config.context.device_id = args_.device_id
    if args_.seed is not None:
        config.seed = args_.seed
    if args_.use_parallel is not None:
        config.use_parallel = args_.use_parallel
    if args_.profile is not None:
        config.profile = args_.profile
    if args_.options is not None:
        config.merge_from_dict(args_.options)

    main(config)
