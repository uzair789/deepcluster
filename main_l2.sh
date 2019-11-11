# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/media/Diana/Data/imagenet2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=100000
WORKERS=12
EXP="/media/Diana/rantao/deepcluster/exp_l2_K100000"
# PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES='2' python main_l2softmax.py ${DIR} --normalize --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} 2>&1 | tee ${EXP}/log.txt
