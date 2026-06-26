import torch
import torch.nn as nn
import torch.nn.functional as F
from mynet.CFE.chennelAttention import *
from mynet.CFE.RPN import *
from mynet.CFE.crossformer import *

class MacBlock(nn.Module):
    def __init__(self, in_channels,out_channels,img_size,patch_size=8):
        super(MacBlock, self).__init__()
        # self.img_size=img_size
        self.ca = CustomBlock(in_channels)
        # self.rpn = RegionConvolution(in_channels, out_channels)
        self.cross = Crossformer(out_channels,img_size=img_size, patch_size=patch_size,in_chans=in_channels)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)
    def forward(self, x1,x2):

        x = x1+x2
        res=x
        x1 = self.ca(x)
        img_size = x.shape[2:]
        # x2 = self.rpn(x,img_size)
        x3 = self.cross(x1,x2)
        # print("x3",x3.shape)
        x3 = self.up(x3)

        # print("x1=",x1.shape)
        # print("x2=",x2.shape)
        # print("x3=",x3.shape)

        out = x1+x2+x3+res
        return out

# 测试
# x1 = torch.randn(1, 512, 64, 64)  # (batch, C, H, W)
# x2 = torch.randn(1, 512, 64, 64)
# model = MacBlock(512,256,64)
# output = model(x1,x2)
# print(output.shape)  # 输出应与输入形状相同 (1, 64, 32, 32)
