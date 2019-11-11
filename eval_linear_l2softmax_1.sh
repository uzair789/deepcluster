# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/media/Diana/Data/imagenet2012/"
MODELROOT="/media/Diana/rantao/deepcluster/exp_l2"
MODEL="${MODELROOT}/checkpoint.pth.tar"
EXP="/media/Diana/rantao/deepcluster/exp_l2/linear_classif"


mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES="5" python eval_linear.py --model ${MODEL} --data ${DATA} --conv 5 --lr 0.01 \
  --wd -7 --verbose --exp ${EXP} --workers 12 --l2 2>&1 | tee ${EXP}/log_conv5_l2.txt

