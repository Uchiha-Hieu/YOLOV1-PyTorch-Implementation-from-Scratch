import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self,in_c,out_c,k=3,s=1,p=1,use_bn=True,use_relu=True):
        """
        in_c : in_channels for conv2d
        out_c : out_channels for conv2d
        k : kernel_size for conv2d
        s : stride for conv2d
        p : padding for conv2d
        use_bn : True if use batchnorm2d
        use_relu : True if use relu
        """
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=k,stride=s,padding=p)
        self.bn = None
        self.relu = None
        if use_bn:
            self.bn = nn.BatchNorm2d(out_c)
        
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        
        if self.relu:
            out = self.relu(out)
        
        return out