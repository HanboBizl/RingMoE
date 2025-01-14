import os
import gc
import cv2
import json
import numpy as np
from osgeo import gdal, gdalconst

def find_tiff_files(directory):
    tiff_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tiff'):
                print("tiff_path:",os.path.join(root, file))
                tiff_files.append(os.path.join(root, file))
    return tiff_files

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
        img_data = np.expand_dims(img_ds.GetRasterBand(1).ReadAsArray(), axis=0)
    if bands > 1:
        for band in range(bands):
            img_data[band, :, :] = img_ds.GetRasterBand(band + 1).ReadAsArray()
    del img_ds
    return img_data


def empty_pixels_exceed_threshold(image, threshold=0.15):
    total_pixels = np.prod(image.shape)
    empty_pixels = np.sum(image == 0)
    empty_pixels_proportion = empty_pixels / total_pixels
    print("empty_pixels_proportion:",empty_pixels_proportion)

    if empty_pixels_proportion < threshold:
        return True
    else:
        return False


def crop_image(image_array, size=192, step=64):
    num_channels, rows, cols = image_array.shape
    print()
    max_r = rows - size
    max_c = cols - size
    cropped_images = []

    # 计算所有可能的裁剪位置
    row_positions = list(range(0, max_r + 1, step))
    col_positions = list(range(0, max_c + 1, step))

    i = 0
    k = 0
    # 遍历所有裁剪位置
    for r in row_positions:
        for c in col_positions:
            i+=1
            # 裁剪图像并将结果添加到列表中
            cropped_image = image_array[:, r:r + size, c:c + size]
            if empty_pixels_exceed_threshold(cropped_image):
                cropped_images.append(cropped_image)
            else:
                k+=1
                print("empty_pixels_too_much")
    print("empty_images:", k/i)
    return cropped_images

def convert_to_8bits(imgs, ratio=0.001):
    transed_images = []
    for src_img in imgs:
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
        transed_images.append(res_img)
    return transed_images


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
        else:
            im_bands = 1
            im_height, im_width = img.shape

        driver = gdal.GetDriverByName('GTIFF')  # 数据类型（必须给定）
        dataset = driver.Create(output_path, im_width, im_height, im_bands, datatype)
        # 逐波段写入数据
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(img[0])
        else:
            for im_band in range(im_bands):
                dataset.GetRasterBand(im_band + 1).WriteArray(img[im_band])


img_filepath = "/mnt/data/sar_l2/SAR_L2_big/"
output_dir = "/mnt/data/cuted_img/SAR_L2_big/"

# 将所有原图路径存在json文件中
SAR_list_L2 = find_tiff_files(img_filepath)
print(SAR_list_L2)
with open("SAR_L2_big_dirList.json",'w') as wf:
    json.dump(SAR_list_L2,wf,indent=4)

with open("SAR_L2_big_dirList.json",'r') as f:
    SAR_L2_tiflist = json.load(f)

processed_tiff = []
img_nums = 0
for file in SAR_L2_tiflist:
    tmp_img = image_read(file)
    cuted_imgs = crop_image(tmp_img,192,64)
    cuted_imgs = convert_to_8bits(cuted_imgs)
    base_name = os.path.basename(file)
    if img_nums % 100 == 0:
        output_path = os.path.join(output_dir, "SAR_L2_" + str(img_nums // 100))
        print("new_path:", output_path)

    image_write(cuted_imgs, output_path, base_name)
    img_nums += 1
    processed_tiff.append(file)
    del tmp_img
    del cuted_imgs
    gc.collect()

with open("sar_l2_big_processed_tif.json",'w') as wf:
    json.dump(processed_tiff,wf,indent=4)
