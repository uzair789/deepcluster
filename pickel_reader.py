import pickle

import argparse
import os
import pickle
import time

#import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--scale', type=float, default=30, help='the scale for l2-softmax')
parser.add_argument('--normalize', action='store_true', help='l2-softmax or not')
parser.add_argument('--cluster_file', type=str, help='path to the cluster binary')

def get_groundtruth_list(dataloader):
    I_list = [[] for x in range(1000)]
    for i, (input_tensor, target) in enumerate(dataloader):
        I_list[target].append(i)
    return I_list

global args


# load the data
end = time.time()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
tra = [transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize]

data_dir = '/home/biometrics/deepcluster/Data/imagenet2012/train'
dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose(tra))
#if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,#args.batch,
                                         num_workers=1,#args.workers,
                                         pin_memory=True)

# load the clusters for the original data
start = time.time()
print("Loading groundtruth Labels")
groundtruth_imagelist = get_groundtruth_list(dataloader)


print('Groundtruth Loaded!!!')
print("TIme to load GT : " +str(round(time.time()-start,4)) + " seconds \n" )


args = parser.parse_args()
print('----**----')
print(args.cluster_file)
pickle_in = open(args.cluster_file,'rb')
log_data = pickle.load(pickle_in)
for i, data_epoch in enumerate(log_data):
    nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(groundtruth_imagelist),
                    clustering.arrange_clustering(data_epoch))
    print('The NMI is epoch ',i, ' NMI = ', nmi )

"""

multiple_cluster_paths = ['results/exp_l2_K5950/clusters', 'results/exp_l2_K59500/clusters']
for i,cluster_path in enumerate(multiple_cluster_paths):
    pickle_in = open(cluster_path, 'rb')
    log_data = pickle.load(pickle_in)
    print('Loaded ', cluster_path)

    f = open('log'+str(i)+'.txt','w')
    f.write(cluster_path+'\n')	
    for i, data_epoch in enumerate(log_data):
        nmi = normalized_mutual_info_score(
                        clustering.arrange_clustering(groundtruth_imagelist),
                        clustering.arrange_clustering(data_epoch))
        line = 'The NMI is epoch '+ str(i) + ' NMI = '+ str(nmi)+'\n'
        f.write(line)


"""
