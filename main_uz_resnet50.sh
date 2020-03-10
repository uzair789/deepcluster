# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/biometrics/deepcluster-git/deepcluster/Data/imagenet2012/train"
ARCH="resnet50"
#ARCH="alexnet"
LR=0.05
WD=-5
K=1000
WORKERS=12
EXP="/home/biometrics/deepcluster-git/deepcluster/exp_uz_resnet_K1000_seqFix"
# PYTHON="/private/home/${USER}/test/conda/bin/python"
CHECKPOINT='/home/biometrics/deepcluster-git/deepcluster/exp_uz_resnet_K1000_seqFix/checkpoints/checkpoint_189_38.0.pth.tar'

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES='0,1,2,3' python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --resume ${CHECKPOINT} --workers ${WORKERS} 2>&1 | tee ${EXP}/log_resume.txt
