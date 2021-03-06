# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/media/Diana/Data/imagenet2012/train"
ARCH="vgg16"
LR=0.05
WD=-5
K=10000
WORKERS=12
EXP="/media/Diana/rantao/deepcluster/exp_vgg16"
# PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES="6,7" python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} 2>&1 | tee ${EXP}/log.txt
