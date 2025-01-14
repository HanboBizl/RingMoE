# The code for RingMoE pre-training and downstream benchmark evaluation
## RingMoE pre-training
### 1. Dependencies
- Mindspore 2.1.0
- Python 3.7
- CANN 6.3.RC2
- Ascend: Ascend910b Cluster
### 2. Datasets
- RingMOSS: https://github.com/HanboBizl/RingMoEDatasets
- Prepare data according to `.ringmoe_framework/datasets/pretrain_dataset.py`
- During pre-training, the data was cropped to a size of 192×192. For detailed operations, please refer to `.ringmoe_framework/datasets/tools/cut_image.py`.

Notably, the data has been converted into the MindRecord format to accelerate data loading.
If needed, you can refer to the official MindSpore documentation for instructions on converting custom datasets into the [MindRecord](https://www.mindspore.cn/docs/zh-CN/r2.4.10/api_python/mindspore.mindrecord.html) format.
### 3. Pre-training
- Distributed training: `sh ./scripts/pre-train_distribute.sh RANK_TABLE_FILE CONFIG_PATH`
- The rank table launch method is exclusive to the Ascend hardware platform.
For details, please refer to the official documentation at [rank table](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/rank_table.html#%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1).
- Here, we also provide reference configurations for 2-node, 8-node, and 16-node setups: `./rank_table_2pcs.json`; `./rank_table_8pcs.json`; `./rank_table_16pcs.json`
- Pre-training the 14.7-billion parameter RingMoE model requires a minimum of 64 Ascend 910B nodes for initialization.

## RingMoE downstream benchmark evaluation
### 1. Scene classification
- The experiments are performed in the [mmpretrain framework](https://github.com/open-mmlab/mmpretrain).
### 2. Semantic segmentation
- The experiments are performed in the [mmsegmentation framework](https://github.com/open-mmlab/mmsegmentation).
### 3. Object detection
- For horizontal object detection, the experiments are performed in the [mmdetection framework](https://github.com/open-mmlab/mmdetection).
- For rotated object detection, the experiments are performed in the [mmrotate framework](https://github.com/open-mmlab/mmrotate).
### 4. Object tracking
- All object tracking tasks utilize the [mmdetection framework](https://github.com/open-mmlab/mmdetection), with [ByteTrack](https://github.com/ifzhang/ByteTrack) as the tracking algorithm.
### 5. Change detection
- The experiments are performed in the [Bidirectional Integration Transformer (BIT) framework](https://github.com/justchenhao/BIT_CD).
### 6. Depth estimation
- The experiments are performed in the [Binsformer framework](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox).