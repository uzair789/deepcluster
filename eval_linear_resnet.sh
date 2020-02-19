# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/home/biometrics/deepcluster-git/deepcluster/Data/imagenet2012/"
MODELROOT="/home/biometrics/deepcluster-git/deepcluster/results/exp_uz_resnet_K1000"
MODEL="${MODELROOT}/checkpoint.pth.tar"
EXP="${MODELROOT}/linear_classif"


mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES="0" python eval_linear_resnet.py --model ${MODEL} --data ${DATA} --conv 100 --lr 0.01 \
  --wd -7 --verbose --exp ${EXP}  --workers 12 2>&1 | tee ${EXP}/log_conv1.txt

