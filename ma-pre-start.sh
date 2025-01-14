echo ***********ma-pre-start*********
#export ASCEND_GLOBAL_LOG_LEVEL=0
#export ASCEND_GLOBAL_EVENT_ENABLE=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=0
# export HCCL_ALGO="level0:fullmesh;level1:fullmesh"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export HCCL_CONNECT_TIMEOUT=6000
export GLOG_logtostderr=0
export GLOG_log_dir=/var/log/npu/slog/ms_info/
export GLOG_v=1
export GLOG_stderrthreshold=2
echo $ASCEND_GLOBAL_LOG_LEVEL
echo $ASCEND_SLOG_PRINT_TO_STDOUT
echo $GLOG_v
npu-smi info
#cd /home/work/user-job-dir/pretrain/
#python pretrain.py --config config/mae-vit-base-p16.yaml --use_parallel False

#python -c "import mindspore;mindspore.run_check()"
cd /home/ma-user/modelarts/user-job-dir/ringmo-framework-ms2.1/
pip install aicc_tools-0.2.1-py3-none-any.whl --ignore-installed
#python ringmo_framework/datasets/tools/mindrecord.py
#python cut_tif.py
#pip install scipy==1.5.2 --ignore-installed --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install mindinsight-1.7.0-py3-none-any.whl --ignore-installed
echo ***********ma-pre-end*********
