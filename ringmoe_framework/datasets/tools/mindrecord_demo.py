import os
from PIL import Image
from io import BytesIO

import mindspore.mindrecord as record


# # 输出的MindSpore Record文件完整路径
MINDRECORD_FILE = "/home/ma-user/work/test.mindrecord"
#
if os.path.exists(MINDRECORD_FILE):
    os.remove(MINDRECORD_FILE)
    os.remove(MINDRECORD_FILE + ".db")

# 定义包含的字段
cv_schema = {"file_name": {"type": "string"},
             "label": {"type": "int32"},
             "data": {"type": "bytes"}}

# 声明MindSpore Record文件格式
writer = record.FileWriter(file_name=MINDRECORD_FILE, shard_num=1)
writer.add_schema(cv_schema, "it is a cv dataset")
writer.add_index(["file_name", "label"])

# 创建数据集
data = []
for i in range(100):
    i += 1
    sample = {}
    white_io = BytesIO()
    Image.new('RGB', (i*10, i*10), (255, 255, 255)).save(white_io, 'JPEG')
    image_bytes = white_io.getvalue()
    sample['file_name'] = str(i) + ".jpg"
    sample['label'] = i
    sample['data'] = white_io.getvalue()

    data.append(sample)
    if i % 10 == 0:
        writer.write_raw_data(data)
        data = []

if data:
    writer.write_raw_data(data)

writer.commit()
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# 读取MindSpore Record文件格式
data_set = ds.MindDataset(dataset_files=MINDRECORD_FILE)
decode_op = vision.Decode()
data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)
count= 0
for item in data_set.create_dict_iterator(output_numpy=True):
    # print("sample: {}".format(item))
    print(item['data'].shape)
    count += 1
print("Got {} samples".format(count))
# # 样本计数
# print("Got {} samples".format(data_set.get_dataset_size()))

