# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from ZJW.mynet.CE.CEBlock import *
from ZJW.mynet.MSSBlock.MSS_res import *
from ZJW.mynet.MAC.macblock import *
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, img_size,dim=256,bilinear=True):
        super().__init__()
        # if bilinear:
        #         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        self.up = nn.ConvTranspose2d(in_channels , in_channels//2 , kernel_size=2, stride=2)
        self.ce = CEBlock(in_channels//2,out_channels,img_size)
        self.mss = Fusion(dim)
        self.mac = MacBlock(in_channels//2,out_channels,img_size)
        self.conv = nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,padding=1)
        self.up1 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(dim, dim * 2, 3, padding=1)

    def forward(self, x1, x2,x3,x4,x5):#x1:上级解码器(y)//x2，x3，x4：第二三四级编码器(z)//x5：同级编码器(x)
        de=self.up(x1)
        mss=self.mss(x2,x3,x4)
        # print("mss=",mss.shape)
        if x5.size(3) < 128:
            mss = self.down1(mss)
            mss = self.conv1(mss)
        elif x5.size(3) > 128:
            mss = self.up1(mss)
        # print("x5=",x5.shape)
        # print("mss=",mss.shape)
        skip=self.mac(x5,mss)
        # diffY = torch.tensor([skip.size()[2] - de.size()[2]])
        # diffX = torch.tensor([skip.size()[3] - de.size()[3]])
        #
        # de = F.pad(de, [diffX // 2, diffX - diffX // 2,
        #                         diffY // 2, diffY - diffY // 2])
        # print("de=",de.shape)
        out=self.ce(skip,de,mss)
        return self.conv(out)




# class Up(nn.Module):
#     """Upscaling then double conv"""
#
#     def __init__(self, in_channels, out_channels, img_size,bilinear=True):
#         super().__init__()
#
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
#
#         self.conv = CEBlock(in_channels, out_channels,img_size=img_size)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
class convup(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)