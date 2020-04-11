# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/home/biometrics/deepcluster-git/deepcluster/Data/imagenet2012/"
MODELROOT="/home/biometrics/deepcluster-git/deepcluster/exp_uz_resnet_K1000_seqFix"
MODEL="${MODELROOT}/checkpoint.pth.tar"
#MODEL="${MODELROOT}/checkpoints/checkpoint_99_20.0.pth.tar"
#EXP="${MODELROOT}/linear_classif_layer3_50176feat"
EXP="${MODELROOT}/linear_classif_layer3EngPoolstride4"


mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES="7" python eval_linear_resnet.py --model ${MODEL} --data ${DATA} --conv 100 --lr 0.01 \
  --wd -7 --verbose --exp ${EXP}  --workers 12 2>&1 | tee ${EXP}/log_conv_layer3stride4.txt

