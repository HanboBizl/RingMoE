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
"""mindrecord of ringmo_framework"""
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
        self.data = data

    def __getitem__(self, index):
        with open(self.data[index], 'rb') as f:
            try:
                img_1 = f.read()
            # pylint: disable=W0703
            except Exception as e:
                print(e)

        row = {"image": img_1}

        try:
            writer.write_raw_data([row])
        # pylint: disable=W0703
        except Exception as e:
            print(e)
        return (np.array([0]),)

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    # 输出的MindSpore Record文件完整路径
    # MINDRECORD_FILE = "/home/ma-user/work/nwpu_aircas/mindrecord_demo"
    # if os.path.exists(MINDRECORD_FILE):
    #     os.remove(MINDRECORD_FILE)
    # else:
    #     os.makedirs(MINDRECORD_FILE, exist_ok=True)
    #     print(f"makdir {MINDRECORD_FILE}")
    # MINDRECORD_FILE = os.path.join(MINDRECORD_FILE, 'aircas.mindrecord')
    #
    # # 定义包含的字段
    # cv_schema = {"image": {"type": "bytes"}}
    #
    # # 声明MindSpore Record文件格式
    # writer = record.FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
    # writer.add_schema(cv_schema, "aircas")
    # writer.set_page_size(1 << 26)
    # ds = de.GeneratorDataset(
    #     source=DataLoader("nwpu_ids.json", data_dir="/home/ma-user/work/nwpu_aircas/imgs/"),
    #     column_names=["image"], shuffle=False,
    #     num_parallel_workers=16, python_multiprocessing=False)
    #
    # index_ = 0
    # t0 = time.time()
    # ds_it = ds.create_dict_iterator()
    # for d in ds_it:
    #     if index_ % 100 == 0:
    #         print(index_)
    #         print(d)
    #     index_ += 1
    #
    # writer.commit()
    # print(time.time() - t0)
    import mindspore.dataset as ds

    file_name = ['/home/ma-user/work/obs/aircas.mindrecord00']
    # 创建MindDataset
    define_data_set = ds.MindDataset(dataset_files=file_name, columns_list=["opt_image"])
    # 创建字典迭代器并通过迭代器读取数据记录
    count = 0
    for item in define_data_set.create_dict_iterator(output_numpy=True):
        print("sample: {}".format(item))
        print(item['image'].shape)
        count += 1
    print("Got {} samples".format(count))
