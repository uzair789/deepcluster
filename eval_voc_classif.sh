# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

VOCDIR="/media/Diana/Data/VOC2007"
MODELROOT="/media/Diana/rantao/deepcluster/exp"
MODEL="${MODELROOT}/checkpoint.pth.tar"

# with training the batch norm
# 72.0 mAP
CUDA_VISIBLE_DEVICES="2" python eval_voc_classif.py --vocdir $VOCDIR --model $MODEL --split trainval --fc6_8 0 --train_batchnorm 1 2>&1 | tee $MODELROOT/log_voc_classify_all_l2_bn.txt

# without training the batch norm
# 70.4 mAP
# CUDA_VISIBLE_DEVICES="2" python eval_voc_classif.py --vocdir $VOCDIR --model $MODEL --split trainval --fc6_8 1 --train_batchnorm 0
