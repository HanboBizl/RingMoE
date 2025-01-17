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
"""mindrecord of ringmoe_framework"""
import json
import os
import time

import mindspore.dataset as de
import mindspore.mindrecord as record
import numpy as np

big_list = []


class DataLoader:
    """data loader"""

    def __init__(self, imgs_path, data_dir=None):
        """Loading image files as a dataset generator."""
        imgs_path = os.path.join(data_dir, imgs_path)
        assert os.path.exists(imgs_path), "imgs_path should be real path:{}.".format(imgs_path)
        with open(imgs_path, 'r') as f:
            data = json.load(f)
        if data_dir is not None:
            data = [os.path.join(data_dir, item) for item in data]
        self.data_1 = data[:len(data)//2]
        self.data_2 = data[len(data) // 2:]

    def __getitem__(self, index):
        with open(self.data_1[index], 'rb') as f:
            try:
                img_1 = f.read()
            # pylint: disable=W0703
            except Exception as e:
                print(e)

        with open(self.data_2[index], 'rb') as f:
            try:
                img_2 = f.read()
            # pylint: disable=W0703
            except Exception as e:
                print(e)

        row = {"opt_image": img_1, "sar_image": img_2, "modal_num": 1}

        try:
            writer.write_raw_data([row])
        # pylint: disable=W0703
        except Exception as e:
            print(e)
        return (np.array([0]),)

    def __len__(self):
        return len(self.data_1)


class ImageLoader:
    """ringmo_framework loader"""

    def __init__(self, opt_ids, sar_ids=None, data_dir=None):
        """Loading image files as a dataset generator."""
        opt_path = os.path.join(data_dir, opt_ids)
        sar_data = None
        with open(opt_path, 'r') as f_opt:
            opt_data = json.load(f_opt)
        if sar_ids:
            sar_path = os.path.join(data_dir, sar_ids)
            with open(sar_path, 'r') as f_sar:
                sar_data = json.load(f_sar)
            if len(opt_data) != len(sar_data):
                raise ValueError("optical image numbers should be equal to sar image numbers.")
        if data_dir is not None:
            opt_data = [os.path.join(data_dir, item) for item in opt_data]
            if sar_ids:
                sar_data = [os.path.join(data_dir, item) for item in sar_data]

        self.opt_data = opt_data
        self.sar_data = sar_data

    def __getitem__(self, index):
        with open(self.opt_data[index], 'rb') as f:
            try:
                opt_img = f.read()
            # pylint: disable=W0703
            except Exception as e:
                print(e)
        if self.sar_data:
            with open(self.sar_data[index], 'rb') as f:
                try:
                    sar_img = f.read()
                # pylint: disable=W0703
                except Exception as e:
                    print(e)
        row = {"opt_image": opt_img, "sar_image": sar_img, "modal_num": 2}

        try:
            writer.write_raw_data([row])
        # pylint: disable=W0703
        except Exception as e:
            print(e)
        return (np.array([0]),)

    def __len__(self):
        return len(self.opt_data)


if __name__ == "__main__":
    # 输出的MindSpore Record文件完整路径
    MINDRECORD_FILE = "/mnt/aircas/pretrain/aircas_opt_mm/aircas_448_450w_record_v3"
    if os.path.exists(MINDRECORD_FILE):
        os.remove(MINDRECORD_FILE)
    else:
        os.makedirs(MINDRECORD_FILE, exist_ok=True)
        print(f"makdir {MINDRECORD_FILE}")
    MINDRECORD_FILE = os.path.join(MINDRECORD_FILE, 'aircas.mindrecord')

    # 定义包含的字段
    cv_schema = {"opt_image": {"type": "bytes"}, "sar_image": {"type": "bytes"}, "modal_num": {"type": "int32"}}

    # 声明MindSpore Record文件格式
    writer = record.FileWriter(file_name=MINDRECORD_FILE, shard_num=20)
    writer.add_schema(cv_schema, "aircas")
    writer.set_page_size(1 << 26)
    ds = de.GeneratorDataset(
        source=DataLoader("/mnt/aircas/pretrain/cuted_images_448/pretrain_ids.json", data_dir="/mnt/aircas/pretrain/cuted_images_448"),
        column_names=["image"], shuffle=False,
        num_parallel_workers=16, python_multiprocessing=False)

    index_ = 0
    t0 = time.time()
    ds_it = ds.create_dict_iterator()
    for d in ds_it:
        if index_ % 100 == 0:
            print(index_)
        index_ += 1

    writer.commit()
    print(time.time() - t0)
