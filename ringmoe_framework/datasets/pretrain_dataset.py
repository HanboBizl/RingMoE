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
"""create train or eval dataset."""

import json
import os
from PIL import Image
import numpy as np
import cv2
from mindspore import context
import mindspore as ms
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.transforms.c_transforms import Compose
from mindspore.dataset.transforms import py_transforms
from mindspore.dataset.vision import Inter
import io
import tifffile as tf
from .utils import _check_pretrain_dataset_config
from .mask.mask_policy import MaskPolicyForSim, MaskPolicyForMae, \
    MaskPolicyForRingMoMM, MaskPolicyForPIMask

MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
MEAN_MS = [0.485 * 255, 0.456 * 255, 0.406 * 255, 0.406 * 255]
STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]
STD_MS = [0.229 * 255, 0.224 * 255, 0.225 * 255, 0.225 * 255]

def get_dataset_size(dataset):
    size = 0
    for _ in dataset.create_dict_iterator():
        size += 1
    return size

class MM_ImageLoader:
    def __init__(self, modal_ids=None,  data_dir=None, modal_type=None):
        """Loading image files as a dataset generator."""
        modal_path = modal_ids

        with open(modal_path, 'r') as f_modal:
            modal_data = json.load(f_modal)

        if data_dir is not None:
            data = [os.path.join(data_dir, item) for item in modal_data]

        self.data = data
        self.modal_type = modal_type
    def __getitem__(self, index):
        out = ()
        if self.modal_type =="opt":
            img = Image.open(self.data[index]).convert("RGB")
        elif self.modal_type =="sar":
            img = Image.open(self.data[index]).convert("RGB")
        elif self.modal_type == "hsi":
            img = Image.open(self.data[index]).convert("RGB")
        else:
            img = Image.open(self.data[index]).convert("RGB")
        out = out + (img,)  #tuple

        return out  # (opt,sar)

    def __len__(self):
        return len(self.data)

class Tiff_converter(py_transforms.PyTensorOperation):
    """Mask generator for simmin arch."""


    def __call__(self, img):
        bytes_data = bytes(img)
        image_data = tf.imread(io.BytesIO(bytes_data))
        return image_data

class Tiff_converter_toRGB(py_transforms.PyTensorOperation):
    """Mask generator for simmin arch."""


    def __call__(self, img):
        bytes_data = bytes(img)
        image_data = tf.imread(io.BytesIO(bytes_data))
        img = np.stack((image_data,image_data,image_data),2)
        return img

class convert_to_8bits(py_transforms.PyTensorOperation):
    """Mask generator for simmin arch."""


    def __call__(self, src_img):
        ratio = 0.001

        # 将图像数据归一化到 0 到 255 的范围
        normalized_img = cv2.normalize(src_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        res_img = np.zeros_like(normalized_img, dtype=np.uint8)
        for channel in range(normalized_img.shape[0]):
            channel_img = normalized_img[channel]
            hist = cv2.calcHist([channel_img], [0], None, [256], [0, 256])  # 修改直方图 bin 的数量为 256
            pixels = channel_img.size
            cum_hist = hist.cumsum(0)
            small_cum = ratio * pixels
            high_cum = pixels - small_cum
            smallValue = np.where(cum_hist > small_cum)[0][0]
            highValue = np.where(cum_hist > high_cum)[0][0]
            if highValue == smallValue:
                res_img[channel] = np.uint8(res_img[channel])
                continue
            channel_img = np.where(channel_img > highValue, highValue, channel_img)
            channel_img = np.where(channel_img < smallValue, smallValue, channel_img)
            scaleRatio = 255.0 / (highValue - smallValue)
            channel_img = channel_img - smallValue
            res_img[channel] = channel_img * scaleRatio
            res_img[channel] = np.uint8(res_img[channel])
        return res_img


class Power_generation(py_transforms.PyTensorOperation):
    """Mask generator for simmin arch."""


    def __call__(self, img):
        # print(img[0]-img[1])
        power = 0
        for i in range(img.shape[0] // 2):
            S_tmp = (np.abs(img[2 * i] + 1j * img[2 * i + 1])) ** 2  # a^2+b^2
            power += S_tmp
        power[power > 1] = 1
        # max = power.max()*0.95
        # min = power.min()*1.05
        power = (power - power.min()) / (power.max() - power.min())  # H*W 归一化0-1值
        power = np.expand_dims(power,axis=0)
        img = np.concatenate((power,img),axis=0)

        return img

class ImageLoader:
    def __init__(self, opt_ids, sar_ids=None, data_dir=None):
        """Loading image files as a dataset generator."""
        opt_path = os.path.join(data_dir, opt_ids)
        print(opt_path)
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
        out = ()
        opt_img = Image.open(self.opt_data[index]).convert("RGB")   #
        out = out + (opt_img,)  #tuple
        if self.sar_data:
            sar_img = Image.open(self.sar_data[index]).convert("RGB")
            out = out + (sar_img,)
        return out  # (opt,sar)

    def __len__(self):
        return len(self.opt_data)


def build_dataset(args):
    if args.input_columns is None:
        args.input_columns = ["image"]

    is_data_parallel = context.get_auto_parallel_context(
        "parallel_mode") == context.ParallelMode.DATA_PARALLEL
    full_batch = context.get_auto_parallel_context("full_batch")
    data_type = args.data_type.lower()
    modal_type = args.modal_type.lower()

    if is_data_parallel or not full_batch:
        if data_type == "mindrecord":
            home_path = os.path.join(os.getcwd(), args.data_path)
            files = os.listdir(args.data_path)
            data_list = [
                os.path.join(home_path, name) for name in files
                if not name.endswith(".db")
            ]
            # Ensure the order of mindrecords is same in all machines, otherwise it will meet loss converge problem.
            data_list.sort()
            dataset = de.MindDataset(
                data_list, columns_list=args.input_columns, num_shards=args.device_num,
                shard_id=args.local_rank, shuffle=args.shuffle,
                num_parallel_workers=args.num_workers,
                num_samples=args.num_samples)
            if args.data_path_sar:
                data_path = os.path.join(args.data_path_sar, args.image_ids)
                dataset_sar = de.MindDataset(
                    data_path, columns_list=args.input_columns, num_shards=args.device_num,
                    shard_id=args.local_rank, shuffle=args.shuffle,
                    num_parallel_workers=args.num_workers,
                    num_samples=args.num_samples)
                dataset = dataset + dataset_sar

        elif data_type == "custom":
            if modal_type == "multi_modal":
                modal_data_paths = args.modal_data_paths
                modal_data_ids = args.modal_data_ids
                dataset = []
                for data_key in modal_data_paths.keys():
                    modal_dataset = de.GeneratorDataset(
                                source=MM_ImageLoader(modal_ids=modal_data_ids[data_key], data_dir=modal_data_paths[data_key], modal_type= data_key),
                                column_names="image",num_shards=args.device_num,
                                shard_id=args.local_rank, shuffle=args.shuffle,
                                num_parallel_workers=args.num_workers,
                                python_multiprocessing=args.python_multiprocessing)
                    dataset.append(modal_dataset)
            else:
                dataset = de.GeneratorDataset(
                    source=ImageLoader(args.image_ids, sar_ids=args.sar_ids, data_dir=args.data_path),
                    column_names=args.input_columns, num_shards=args.device_num,
                    shard_id=args.local_rank, shuffle=args.shuffle,
                    num_parallel_workers=args.num_workers,
                    python_multiprocessing=args.python_multiprocessing)
        else:
            raise NotImplementedError("Only support mindrecord or custom mode,but get {}".format(data_type))

    elif full_batch:
        if data_type == "mindrecord":
            if modal_type == "multi_modal":
                modal_data_paths = []
                if os.path.exists(os.path.join(args.modal_data_paths, 'opt')):
                    opt_path = os.path.join(args.modal_data_paths, 'opt')
                else:
                    opt_path = os.path.join(args.modal_data_paths, 'data_url_0')

                if os.path.exists(os.path.join(args.modal_data_paths, 'SAR_L1')):
                    sar1_path = os.path.join(args.modal_data_paths, 'SAR_L2')
                else:
                    sar1_path = os.path.join(args.modal_data_paths, 'data_url_1')

                if os.path.exists(os.path.join(args.modal_data_paths, 'SAR_L2')):
                    sar2_path = os.path.join(args.modal_data_paths, 'SAR_L2')
                else:
                    sar2_path = os.path.join(args.modal_data_paths, 'data_url_2')

                if os.path.exists(os.path.join(args.modal_data_paths, 'MS')):
                    ms_path = os.path.join(args.modal_data_paths, 'MS')
                else:
                    ms_path = os.path.join(args.modal_data_paths, 'data_url_3')
                modal_data_paths.append(opt_path)
                modal_data_paths.append(sar1_path)
                modal_data_paths.append(sar2_path)
                modal_data_paths.append(ms_path)

                dataset = []
                for ids, path  in enumerate(modal_data_paths):
                    home_path = os.path.join(os.getcwd(), path)
                    files = os.listdir(path)
                    data_list = [
                        os.path.join(home_path, name) for name in files
                        if not name.endswith(".db")
                    ]
                    # Ensure the order of mindrecords is same in all machines, otherwise it will meet loss converge problem.
                    data_list.sort()
                    modal_dataset = de.MindDataset(
                        data_list, columns_list=args.input_columns, shuffle=False,
                        num_parallel_workers=args.num_workers,
                        num_samples=args.num_samples)
                    dataset.append(modal_dataset)

            else:

                home_path = os.path.join(os.getcwd(), args.data_path)
                files = os.listdir(args.data_path)
                data_list = [
                    os.path.join(home_path, name) for name in files
                    if not name.endswith(".db")
                ]
                # Ensure the order of mindrecords is same in all machines, otherwise it will meet loss converge problem.
                data_list.sort()
                dataset = de.MindDataset(
                    data_list, columns_list=args.input_columns, shuffle=False,
                    num_parallel_workers=args.num_workers,
                    num_samples=args.num_samples)


                if args.data_path_sar:
                    # print(args.data_path_sar)
                    data_path = os.path.join(args.data_path_sar, args.image_ids)
                    dataset_sar = de.MindDataset(
                        data_path, columns_list=args.input_columns, shuffle=False,
                        num_parallel_workers=args.num_workers,
                        num_samples=args.num_samples)
                    dataset = dataset + dataset_sar

        elif data_type == "custom":
            if modal_type == "multi_modal":
                modal_data_paths = args.modal_data_paths
                modal_data_ids = args.modal_data_ids
                dataset = []
                for data_key in modal_data_paths.keys():
                    modal_dataset = de.GeneratorDataset(
                                source=MM_ImageLoader(modal_ids=modal_data_ids[data_key], data_dir=modal_data_paths[data_key], modal_type= data_key),
                                column_names="image", shuffle=False,
                                num_parallel_workers=args.num_workers,
                                python_multiprocessing=args.python_multiprocessing)
                    dataset.append(modal_dataset)

            else:
                dataset = de.GeneratorDataset(
                    source=ImageLoader(args.image_ids, data_dir=args.data_path),
                    column_names=args.input_columns, shuffle=False,
                    num_parallel_workers=args.num_workers,
                    python_multiprocessing=args.python_multiprocessing)
        else:
            raise NotImplementedError("Only support mindrecord or custom mode,but get {}".format(data_type))
    else:
        raise ValueError("if now is data context mode, full batch should be False.")
    return dataset


def build_transforms(args):
    """build transforms"""
    trans = [
        C.RandomCropDecodeResize(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            # scale=(args.crop_min, 1.0),
            interpolation=Inter.BICUBIC),
        # C.RandomResizedCrop(
        #     args.image_size,
        #     scale=(args.crop_min, 1.0),
        #     interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]
    trans = Compose(trans)
    return trans

def build_transforms_list(args):
    """build transforms"""
    trans_list=[]
    trans_modal1 = [
        C.RandomCropDecodeResize(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            # scale=(args.crop_min, 1.0),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]
    trans_modal2 = [
        C.RandomResizedCrop(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        # C.Normalize(mean=MEAN_MS, std=STD_MS),
        C.HWC2CHW(),
    ]
    trans_modal3 = [
        # C.RandomCropDecodeResize(
        #     args.image_size,
        #     scale=(args.data_scale_min, args.data_scale_max),
        #     # scale=(args.crop_min, 1.0),
        #     interpolation=Inter.BICUBIC),
        C.RandomResizedCrop(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]
    trans_modal4 = [
        C.RandomResizedCrop(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN_MS, std=STD_MS),
        C.HWC2CHW(),
    ]
    trans_list.append(Compose(trans_modal1))
    trans_list.append(Compose(trans_modal2))
    trans_list.append(Compose(trans_modal3))
    trans_list.append(Compose(trans_modal4))
    return trans_list

def build_transforms_list_withoutL1(args): # 0401 transform
    """build transforms"""
    trans_list=[]
    trans_modal1 = [
        C.RandomCropDecodeResize(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            # scale=(args.crop_min, 1.0),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]

    trans_modal3 = [
        # C.RandomCropDecodeResize(
        #     args.image_size,
        #     scale=(args.data_scale_min, args.data_scale_max),
        #     # scale=(args.crop_min, 1.0),
        #     interpolation=Inter.BICUBIC),
        C.RandomResizedCrop(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]
    trans_modal4 = [
        C.RandomResizedCrop(
            args.image_size,
            scale=(args.data_scale_min, args.data_scale_max),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=args.prop),
        C.Normalize(mean=MEAN_MS, std=STD_MS),
        C.HWC2CHW(),
    ]
    trans_list.append(Compose(trans_modal1))
    trans_list.append(Compose(trans_modal3))
    trans_list.append(Compose(trans_modal4))
    return trans_list
def build_mask(args, ds, input_columns=None, output_columns=None):   #0401
    """build mask"""
    batch_size = args.batch_size
    if args.arch == 'simmim':
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask"]
        generate_mask = MaskPolicyForSim(
            input_size=args.image_size, mask_patch_size=args.mask_patch_size,
            model_patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    elif args.arch == 'simmim_moe':
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask"]
        generate_mask = MaskPolicyForSim(
            input_size=args.image_size, mask_patch_size=args.mask_patch_size,
            model_patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    elif args.arch == 'simmim_single_moe':
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask"]
        generate_mask = MaskPolicyForSim(
            input_size=args.image_size, mask_patch_size=args.mask_patch_size,
            model_patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    elif args.arch == 'mae':
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask", "ids_restore", "unmask_index"]
        generate_mask = MaskPolicyForMae(
            input_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    elif args.arch == "ringmo_mm":
        if not input_columns:
            input_columns = ["image1", "image2"]
        if not output_columns:
            output_columns = ["image1", "image2", "mask", "ids_restore"]

        generate_mask = MaskPolicyForRingMoMM(
            input_size=args.image_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
        batch_size = args.batch_size // 2
    elif args.arch == "ringmo":
        if not input_columns:
            input_columns = ["image"]
        if not output_columns:
            output_columns = ["image", "mask"]
        if args.use_lbp:
            output_columns  = ["image", "lbp_image", "mask"]
        generate_mask = MaskPolicyForPIMask(
            input_size=args.image_size, mask_patch_size=args.mask_patch_size,
            mask_ratio=args.mask_ratio, inside_ratio=args.inside_ratio, use_lbp=args.use_lbp)
        ds = ds.map(
            operations=generate_mask, input_columns=input_columns,
            output_columns=output_columns, num_parallel_workers=args.num_workers,
            python_multiprocessing=args.python_multiprocessing)
    else:
        raise NotImplementedError(args.arch)
    ds = ds.batch(batch_size, drop_remainder=True, num_parallel_workers=args.num_workers)
    return ds

def data_generator(masked_datasets, batch_size):
    """
    Generate data batches based on the specified modal ratios and image counts.

    Args:
        masked_datasets (list): A list of masked datasets.
        batch_size (int): The batch size for generating data.
        modal_ratios (list): A list of modal ratios.
        modal_image_counts (list): A list of image counts for each modal.

    Yields:
        tuple: A tuple containing image data, mask data, and modal index.

    Raises:
        ValueError: If the lengths of modal_ratios and modal_image_counts are not the same.

    """
    modal_image_counts = [get_dataset_size(dataset) for dataset in masked_datasets]
    print(modal_image_counts)
    total_images = sum(modal_image_counts)
    modal_ratios = [count / total_images for count in modal_image_counts]

    # Check if the lengths of modal_ratios and modal_image_counts are the same
    if len(modal_ratios) != len(modal_image_counts):
        raise ValueError("The lengths of modal_ratios and modal_image_counts must be the same.")

    # Ensure that the total number of samples in a batch equals batch_size
    modal_counts_in_batch = [int(ratio * batch_size) for ratio in modal_ratios]
    while sum(modal_counts_in_batch) != batch_size:
        if sum(modal_counts_in_batch) > batch_size:
            max_count_idx = modal_counts_in_batch.index(max(modal_counts_in_batch))
            modal_counts_in_batch[max_count_idx] -= 1
        else:
            max_count_idx = modal_counts_in_batch.index(max(modal_counts_in_batch))
            modal_counts_in_batch[max_count_idx] += 1

    max_iter = max([modal_image_counts[i] // modal_counts_in_batch[i] for i in range(len(modal_image_counts))])
    modal_iterators = [iter(dataset.create_dict_iterator()) for dataset in masked_datasets]
    for _ in range(max_iter):
        for idx, count in enumerate(modal_counts_in_batch):
            data_iter = modal_iterators[idx]
            for _ in range(count):
                try:
                    data = next(data_iter)
                except StopIteration:
                    # If the data for a modal is exhausted, start from the beginning
                    modal_iterators[idx] = iter(masked_datasets[idx].create_dict_iterator())
                    data_iter = modal_iterators[idx]
                    data = next(data_iter)
                yield (data['image'].asnumpy(), data['mask'].asnumpy(), idx)

def create_pretrain_dataset(args):
    """Create dataset for self-supervision training."""
    _check_pretrain_dataset_config(args)
    dataset_config = args.pretrain_dataset
    de.config.set_seed(args.seed)
    de.config.set_prefetch_size(dataset_config.prefetch_size)
    de.config.set_numa_enable(dataset_config.numa_enable)
    if args.auto_tune and not args.profile:
        os.makedirs(args.filepath_prefix, exist_ok=True)
        args.filepath_prefix = os.path.join(args.filepath_prefix, "autotune")
        de.config.set_enable_autotune(True, filepath_prefix=args.filepath_prefix)
        de.config.set_autotune_interval(args.autotune_per_step)

    ds = build_dataset(dataset_config)

    if dataset_config.modal_type.lower() == "multi_modal":
        transforms_list = build_transforms_list(dataset_config)
        transformed_datasets = []
        tiff_converter = Tiff_converter()   #SAR_L1/MS
        tiff_converter_RGB = Tiff_converter_toRGB()   #SAR_L2

        for i, dataset in enumerate(ds):
            if i==1:
                dataset = dataset.map(operations=tiff_converter,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.input_columns,
                                      num_parallel_workers=dataset_config.num_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)
                power_generation = Power_generation()  ## SAR_L1
                dataset = dataset.map(operations=power_generation, input_columns=dataset_config.input_columns,
                            output_columns=dataset_config.input_columns,
                            num_parallel_workers=dataset_config.num_workers,
                            python_multiprocessing=dataset_config.python_multiprocessing)

            elif i ==2:
                dataset = dataset.map(operations=tiff_converter_RGB,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.input_columns,
                                      num_parallel_workers=dataset_config.num_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

            elif i==3:
                dataset = dataset.map(operations=tiff_converter,
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.input_columns,
                                      num_parallel_workers=dataset_config.num_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)



            dataset = dataset.map(operations=transforms_list[i],
                                      input_columns=dataset_config.input_columns,
                                      output_columns=dataset_config.input_columns,
                                      num_parallel_workers=dataset_config.num_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)
            transformed_datasets.append(dataset)

        generate_mask = MaskPolicyForSim(
            input_size=dataset_config.image_size, mask_patch_size=dataset_config.mask_patch_size,
            model_patch_size=dataset_config.patch_size, mask_ratio=dataset_config.mask_ratio)

        masked_datasets = [dataset.map(operations=generate_mask,
                                       input_columns=dataset_config.input_columns,
                                       output_columns=dataset_config.output_columns,
                                       num_parallel_workers=dataset_config.num_workers,
                                       python_multiprocessing=dataset_config.python_multiprocessing)
                           for i, dataset in enumerate(transformed_datasets)]
        len_dataset = [get_dataset_size(dataset) for dataset in masked_datasets]
        print('len_dataset', len_dataset)
        max_len = max(len_dataset)
        min_len = min(len_dataset)
        repeated_datasets = [dataset.repeat(max_len // len_dataset[i]) for i, dataset in enumerate(masked_datasets)]
        repeated_datasets = [dataset.rename(input_columns=["image", "mask"], output_columns=["image{}".format(i), "mask{}".format(i)])
            for i, dataset in enumerate(repeated_datasets)]
        zip_dataset = de.ZipDataset(repeated_datasets)
        dataset = zip_dataset.batch(dataset_config.batch_size, drop_remainder=True, num_parallel_workers=args.num_workers)
        dataset = dataset.repeat(dataset_config.epoch)

        return dataset

    else:
        # tiff_converter = Tiff_converter()   #SAR_L1/ MS
        # ds = ds.map(operations=tiff_converter, input_columns=dataset_config.input_columns,
        #     output_columns=dataset_config.input_columns,num_parallel_workers=dataset_config.num_workers,
        #                 python_multiprocessing=dataset_config.python_multiprocessing)

        tiff_converter = Tiff_converter_toRGB()   #SAR_L2
        ds = ds.map(operations=tiff_converter, input_columns=dataset_config.input_columns,
            output_columns=dataset_config.input_columns,num_parallel_workers=dataset_config.num_workers,
                        python_multiprocessing=dataset_config.python_multiprocessing)

        transforms = build_transforms_list(dataset_config)   #
        # transforms = build_transforms(dataset_config)   #
        for column in dataset_config.input_columns:
            if column == "modal_num":
                continue
            ds = ds.map(input_columns=column,
                        operations=transforms[2],   #opt/SAR_l1/SAR_L2/MS
                        num_parallel_workers=dataset_config.num_workers,
                        python_multiprocessing=dataset_config.python_multiprocessing)

        # power_generation = Power_generation()       ## SAR_L1
        # ds = ds.map(operations=power_generation, input_columns=dataset_config.input_columns,
        #     output_columns=dataset_config.input_columns,num_parallel_workers=dataset_config.num_workers,
        #                 python_multiprocessing=dataset_config.python_multiprocessing)

        ds = build_mask(dataset_config, ds,
                        input_columns=dataset_config.input_columns,
                        output_columns=dataset_config.output_columns)
        ds = ds.repeat(dataset_config.epoch)


        return ds

if __name__ == "__main__":
    data_path = '/home/ma-user/work/multi_modal_demo/SAR_L2'
    home_path = os.path.join(os.getcwd(), data_path)
    files = os.listdir(data_path)
    data_list = [
        os.path.join(home_path, name) for name in files
        if not name.endswith(".db")
    ]
    # Ensure the order of mindrecords is same in all machines, otherwise it will meet loss converge problem.
    data_list.sort()
    dataset = de.MindDataset(data_list, columns_list=["data"])
    tiff_converter= Tiff_converter_toRGB()
    dataset = dataset.map(
        operations=tiff_converter, input_columns=["data"],
        output_columns=["data"])
    # converter_8_bits= convert_to_8bits()
    # dataset = dataset.map(
    #     operations=converter_8_bits, input_columns=["data"],
    #     output_columns=["data"])
    trans = [
        # C.Decode(),
        C.RandomResizedCrop(
            192,
            scale=(0.2, 0.5),
            interpolation=Inter.BICUBIC),
        C.RandomHorizontalFlip(prob=0.5),
        C.Normalize(mean=MEAN, std=STD),
        C.HWC2CHW(),
    ]
    trans = Compose(trans)
    dataset = dataset.map(
        operations=trans, input_columns=["data"],
        output_columns=["data"])
    # power_generation = Power_generation()  ## SAR_L1
    # dataset = dataset.map(operations=power_generation, input_columns=["data"])
    # generate_mask=MaskPolicyForSim(
    #     input_size=192, mask_patch_size=32,
    #     model_patch_size=16, mask_ratio=0.6)
    # dataset = dataset.map(
    #     operations=generate_mask, input_columns=["data"],
    #     output_columns=["data","mask"])
    # for data in dataset.create_dict_iterator():
    #     images, mask  = data['data'], data["mask"]
    #     print('Images shape:', images.shape)
    #     print('Images:', ((images>0)&(images<1)).float())
    #     print('mask shape:', mask.shape)
    #     print('Images dtype:', images.dtype)
    for data in dataset.create_dict_iterator():
        images  = data['data']
        print('Images shape:', images.shape)
        print(images)
