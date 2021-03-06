# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/biometrics/deepcluster/Data/imagenet2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=1000
WORKERS=12
EXP="/home/biometrics/deepcluster/exp_uz_l2_K1000_debug"
# PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES='5' python main_l2softmax.py ${DIR} --normalize --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} 2>&1 | tee ${EXP}/log.txt
