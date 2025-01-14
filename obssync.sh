#!/bin/bash
CURRENT_DIR=$(cd `dirname $0`; pwd)
CURRENT_NAME="${CURRENT_DIR##*/}"

# Step1, download obsutil tools
# wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz /root/
# cd /root/
# tar -xzvf obsutil_linux_arm64.tar.gz
# cd obsutil_linux_arm64_5.3.4/
# chmod 755 obsutil

rm -rf rank_0 log kernel_meta summary/*

cd /root/obsutil_linux_arm64_5.3.4/
./obsutil config -i=HC4OBVO4QIJ1BHBSE7QX -k=E89SYVtEpEUv7X7iAeNkNnMBmUovDHJmy8M18pIe -e=https://obs.cn-southwest-228.cdzs.cn
./obsutil ls -s
# obs路径请根据自己的工作目录替换填写
OBS_PATH=obs://linbert/work
./obsutil sync $CURRENT_DIR $OBS_PATH/$CURRENT_NAME

