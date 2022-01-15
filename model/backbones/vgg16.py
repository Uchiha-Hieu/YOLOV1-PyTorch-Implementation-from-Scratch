import torch
import torch.nn as nn
from .components import conv_bn_relu

config =[
    16,16,'M',32,32,'M',64,64,'M',128,128,'M',256,256,'M'
]

class VGG(nn.Module):
    def __init__(self,in_c,config,img_size=224):
        super(VGG,self).__init__()
        self.in_c = in_c
        self.config = config
        self.out_c = config[-2]
        self.out_size = img_size//32
        self.feature_layers = self.make_layers()
    
    def make_layers(self):
        layers = []
        in_c = self.in_c
        for cf in self.config:
            if cf == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]

            else:
                layers += [conv_bn_relu(in_c=in_c,out_c=cf,k=3,s=1,p=1,use_bn=True,use_relu=True)]
                in_c = cf
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        return self.feature_layers(x)

def VGG16(in_c):
    return VGG(in_c,config)