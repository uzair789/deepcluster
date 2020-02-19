import torch
from torchsummary import summary

from models.resnet50 import resnet50





model = resnet50(sobel=True)
model.cuda()
summary(model, (3, 224, 224))

f1 = open('resnet_debug_arch.txt', 'w')
for name, param in model.state_dict().items():
    l = name+'\t'+str(param.size())
    f1.write(l+'\n')

f1.close()
