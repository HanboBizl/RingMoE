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
"""callback of ringmo_framework"""
from .monitor import StateMonitor
import time
import os
import pathlib
import math
import numpy as np
from mindspore import merge_pipeline_strategys, transform_checkpoints
from mindspore.train.callback import Callback
import shutil
from aicc_tools.ailog.log import get_logger
from aicc_tools.utils.utils import LOCAL_DEFAULT_PATH, PROFILE_INFO_PATH
from aicc_tools.utils.utils import get_net_outputs
from aicc_tools.utils.validator import check_obs_url, check_in_modelarts, format_path, Validator
from aicc_tools.utils.monitor import CheckpointMointor, ProfileMonitor

class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
        ValueError: If data_size is not an integer or less than zero.
    """

    def __init__(self, data_size=None, has_trained_epoch=0, has_trained_step=0, log=get_logger()):
        super(LossMonitor, self).__init__()
        if not isinstance(data_size, int) or data_size < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(data_size))
        self._last_print_time = 0
        self._dataset_size = data_size
        self.step_time = time.time()
        self.log = log
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        print("Load the trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)

    def step_begin(self, run_context):
        """
        Record time at the begin of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        self.step_time = time.time()

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()

        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                epoch_num -= 1
            loss = get_net_outputs(cb_params.net_outputs)
            step_seconds = (time.time() - self.step_time) * 1000
            self.log.info('epoch: {} step: {}, loss is {}; per step time: {:5.3f} ms'.format(
                int(epoch_num) + int(self.has_trained_epoch), cb_params.cur_step_num + int(self.has_trained_step), loss, step_seconds))


        # cb_params = run_context.original_args()
        #
        # step_seconds = (time.time() - self.step_time) * 1000
        #
        # loss = get_net_outputs(cb_params.net_outputs)
        #
        # cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        #
        # if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
        #     raise ValueError('epoch: {} step: {}. Invalid loss, terminating training.'.format(
        #         cb_params.cur_epoch_num, cur_step_in_epoch))
        # if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
        #     self._last_print_time = cb_params.cur_step_num
        #     self.log.info('epoch: {} step: {}, loss is {}; per step time: {:5.3f} ms'.format(
        #         cb_params.cur_epoch_num, cur_step_in_epoch, loss, step_seconds))

def checkpoint_monitor(directory=None,local_rank=None, prefix='CKP', **kwargs):
    """Save checkpoint in training for network."""
    # rank_id = int(os.getenv('RANK_ID', '0'))
    if directory:
        directory = os.path.join(directory, 'rank_{}'.format(local_rank))
        directory = os.path.join(directory, 'checkpoint')
    elif directory is None:
        directory = os.path.join(LOCAL_DEFAULT_PATH, 'rank_{}'.format(local_rank))
        directory = os.path.join(directory, 'checkpoint')
    Validator.check_type(directory, str)
    format_path(directory)
    print('obs save_path',directory)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    ckpt_cb = CheckpointMointor(prefix=prefix, directory=directory, **kwargs)
    return ckpt_cb.save_checkpoint()

def build_pretrain_callback(args, cfts):
    """build pretrain callback"""
    train_config = args.train_config
    ckpt_config = args.callback.ckpt_config
    summary_config = args.callback.summary_config
    data_size = args.data_size
    # loss_cb = cfts.loss_monitor(per_print_times=1)
    loss_cb = LossMonitor(data_size=data_size,has_trained_epoch=0, has_trained_step=0,log=get_logger())

    summary_cb = cfts.summary_monitor(**summary_config)
    ckpt_append_info = [
        {"epoch_num": train_config.has_trained_epoches,
         "step_num": train_config.has_trained_steps}
    ]

    if ckpt_config.obs_local_path:
        save_path = os.path.join(ckpt_config.obs_local_path,ckpt_config.prefix)
        ckpt_save_path = os.path.join(save_path,'ckpt')

    else:
        ckpt_save_path =None




    ckpt_cb = checkpoint_monitor(directory=ckpt_save_path, local_rank = args.local_rank, prefix=ckpt_config.prefix + "_rank_{}".format(args.local_rank),
                                      save_checkpoint_steps=ckpt_config.save_checkpoint_steps,
                                      keep_checkpoint_max=ckpt_config.keep_checkpoint_max,
                                      integrated_save=ckpt_config.integrated_save,
                                      async_save=ckpt_config.async_save,
                                      append_info=ckpt_append_info)
    obs_cb = cfts.obs_monitor()
    callback = [loss_cb, ckpt_cb, summary_cb, obs_cb]
    return callback


def build_finetune_callback(args, cfts, eval_engine):
    """build finetune callback"""
    ckpt_config = args.callback.ckpt_config
    train_config = args.train_config
    dataset_config = args.finetune_dataset
    state_cb = StateMonitor(data_size=train_config.per_epoch_size,
                            tot_batch_size=train_config.batch_size * args.device_num,
                            eval_interval=dataset_config.eval_interval,
                            eval_offset=dataset_config.eval_offset,
                            eval_engine=eval_engine,
                            logger=args.logger.info)

    ckpt_append_info = [{"epoch_num": train_config.has_trained_epoches,
                         "step_num": train_config.has_trained_steps}]
    ckpt_cb = cfts.checkpoint_monitor(prefix=ckpt_config.prefix + "_rank_{}".format(args.local_rank),
                                      save_checkpoint_steps=ckpt_config.save_ckpt_epochs * train_config.per_epoch_size,
                                      keep_checkpoint_max=ckpt_config.keep_checkpoint_max,
                                      integrated_save=ckpt_config.integrated_save,
                                      async_save=ckpt_config.async_save,
                                      append_info=ckpt_append_info)
    obs_cb = cfts.obs_monitor()
    callback = [ckpt_cb, state_cb, obs_cb]  #
    return callback
