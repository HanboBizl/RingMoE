import os
import gc
import cv2
import json
import numpy as np
from osgeo import gdal, gdalconst

def get_dirpath(img_path):
    dirList = [name for name in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, name))]
    SAR_L1 = []
    for dir in dirList:
        dir_path = os.path.join(img_path, dir)
        print(dir_path)
        SAR_L1.extend([dir_path])
    return SAR_L1

def crop_image(image_array, size=192, step=64):
    num_channels, rows, cols = image_array.shape
    print()
    max_r = rows - size
    max_c = cols - size
    cropped_images = []

    # 计算所有可能的裁剪位置
    row_positions = list(range(0, max_r + 1, step))
    col_positions = list(range(0, max_c + 1, step))

    # 遍历所有裁剪位置
    for r in row_positions:
        for c in col_positions:
            # 裁剪图像并将结果添加到列表中
            cropped_images.append(image_array[:, r:r + size, c:c + size])

    return cropped_images

def image_read(input_file):
    """
    影像读取
    input_file:输入的影像路径
    返回列数, 行数, 波段数, 投影, 仿射矩阵, 遥感数据数组
    """
    img_ds = gdal.Open(input_file)  # 利用gdal打开影像
    width, height, bands = img_ds.RasterXSize, img_ds.RasterYSize, img_ds.RasterCount  # 获取影像行列和波段数
    proj = img_ds.GetProjection()  # 获取影像投影信息
    geotrans = img_ds.GetGeoTransform()  # 获取仿射矩阵, 六参数
    img_data = np.zeros(shape=(bands, height, width))  # 构建一个多维0数组,用来存放遥感影像值
    ##逐波段读取数据
    if bands == 1:
        img_data[0, :, :] = img_ds.GetRasterBand(1).ReadAsArray()
    if bands > 1:
        for band in range(bands):
            img_data[band, :, :] = img_ds.GetRasterBand(band + 1).ReadAsArray()
    del img_ds
    return img_data

def image_write(im_data, output_file, base_name):
    """
    影像数据写入tif文件
    output_file:输出的影像路径
    im_proj，im_geotrans:输出影像的投影和六参数
    im_data:要写入tif的栅格数据
    """
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    # 判断栅格数据的数据类型
    for i, img in enumerate(im_data):
        base_name = base_name.replace('.tif', '')
        output_path = os.path.join(output_file, f"{base_name}_{i}.tif")
        datatype = gdal.GDT_UInt16  # 修改数据类型为 16 位无符号整数
        # 获取栅格数据的波段数量, 行数和列数
        if len(img.shape) == 3:
            im_bands, im_height, im_width = img.shape
            print("image.shape:",img.shape)
        else:
            im_bands = 1
            im_height, im_width = img.shape

        driver = gdal.GetDriverByName('GTIFF')  # 数据类型（必须给定）
        dataset = driver.Create(output_path, im_width, im_height, im_bands, datatype)
        # 逐波段写入数据
        if im_bands == 1:
            rows = img.shape[1]
            cols = img.shape[2]
            # 创建一个大小为 (rows, cols) 的数组，填充为 1
            img_array = np.ones((rows, cols))
            dataset.GetRasterBand(1).WriteArray(img_array)
        else:
            for im_band in range(im_bands):
                dataset.GetRasterBand(im_band + 1).WriteArray(img[im_band])


img_filepath = "/mnt/obs/SAR_L1_big"
output_dir = "/mnt/data/cuted_img/SAR_L1_big/"
SAR_list_L1 = get_dirpath(img_filepath)
print(SAR_list_L1)
with open("SAR_L1_dirList.json",'w') as wf:
    json.dump(SAR_list_L1,wf,indent=4)

with open("SAR_L1_dirList.json",'r') as f:
    SAR_L1_darlist = json.load(f)

input_files = []
for sar_dir in SAR_L1_darlist:
    for file in os.listdir(sar_dir):
        print("file:",file)
        if 'VV' in file and file.endswith('.tiff'):
            input_files.append(os.path.join(sar_dir,file))

with open("SAR_L1_tifflist.json", 'w') as  wf:
    json.dump(input_files,wf,indent=4)


print("input_file ok")
processed_tiff = []
img_nums = 0
for file in input_files:
    need_cut_tif = []
    print("tiff:",file)
    need_cut_tif.append((file.replace('VV', 'HH')))
    need_cut_tif.append((file.replace('VV', 'HV')))
    need_cut_tif.append((file.replace('VV', 'VH')))
    need_cut_tif.append((file.replace('VV', 'VV')))
    all_tiff = []
    merged_tiff = []
    for tif in need_cut_tif:
        tmp_img = image_read(file)
        cuted_imgs = crop_image(tmp_img,192,64)
        all_tiff.append(cuted_imgs)

    for i in range(len(all_tiff[0])):
        channels_data = []
        for one_tiff in all_tiff:
            channels_data.extend(one_tiff[i])
        merged_image = cv2.merge(channels_data)

        merged_tiff.append(merged_image)
    if img_nums % 100 == 0:
        output_path = os.path.join(output_dir, "SAR_L1_" + str(img_nums // 100))
        print("new_path:", output_path)
    base_name = os.path.basename(file)

    image_write(merged_tiff, output_path, base_name)
    img_nums += 1
    processed_tiff.append(file)
    del need_cut_tif, all_tiff, merged_tiff
    gc.collect()

with open("sar_l1_processed_tif.json",'w') as wf:
    json.dump(processed_tiff,wf,indent=4)
