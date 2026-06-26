import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ops.bra_legacy import BiLevelRoutingAttention
# from biformer import Block
import os
# from skimage.metrics import structural_similarity as ssim
from ZJW.mynet.CE.ssim2 import *

#-------------------------------------------------------------------------------------------------
class fourconv(nn.Module):
    def __init__(self,unit,k,in_channels,out_channels,win_size=11, size_average=True, sigma=1.5):
        super().__init__()
        self.Unit = unit
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding=(0, (k-1)//2))
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=(k,1),padding=((k-1)//2,0))
        # self.win_size = win_size
        # self.size_average = size_average
        # self.sigma = sigma
        # self.channel = in_channels
        self.ssim = GlobalSSIM(win_size=win_size, size_average=size_average, sigma=sigma,in_channels=in_channels)


    def forward(self, x,y): #y来自上一层特征图
            # x的形状为(B, C, H, W)，其中B是批次大小
        B, C,H, W = x.shape

            # 拉伸图像，形状变为(B, C, 1, H*W)
        stretched1 = x.view(B, C, 1, H * W)
        x1 = self.conv1(stretched1) #(1×HW）
        stretched2 = x1.view(B, C, H * W, 1)
        x2 = self.conv2(stretched2)  #（HW×1）
        x2 = torch.rot90(x2,1,[2,3]) #(1*HW)
        x2 = torch.roll(x2, shifts=1*self.Unit, dims=3)
        cropped_part2 = x2[:, :, :, -self.Unit*1:]
        main_part2 = x2[:, :, :, :-self.Unit*1]
        x2 = torch.cat((cropped_part2, main_part2), dim=3)
        rotated1 = torch.rot90(stretched1, 2, [2, 3])
        x3 = self.conv1(rotated1)  #（1×HW）
        x3 = torch.roll(x3, shifts=2*self.Unit, dims=3)
        cropped_part3 = x3[:, :, :, -self.Unit * 2:]
        main_part3 = x3[:, :, :, :-self.Unit * 2]
        x3 = torch.cat((cropped_part3, main_part3), dim=3)
        rotated2 = torch.rot90(stretched2, 2, [2, 3])
        x4 = self.conv2(rotated2) #torch.rot90(x2,1,[2,3])（HW×1）
        x4 = torch.rot90(x4,1,[2,3]) #(1*HW)
        x4 = torch.roll(x4, shifts=3 * self.Unit, dims=3)
        cropped_part4 = x4[:, :, :, -self.Unit * 3:]
        main_part4 = x4[:, :, :, :-self.Unit * 3]
        x4 = torch.cat((cropped_part4, main_part4), dim=3)
        # x = torch.cat((x1,x2,x3,x4),dim=2)
        # p_w1 = F.adaptive_max_pool2d(x, (4, 1))
        # p_cw1 = torch.max(p_w1, dim=1,keepdim=True)[0]
        # p_w2 = F.adaptive_max_pool2d(x, (4,1))
        # p_cw2 = torch.mean(p_w2, dim=1, keepdim=True)
        # p_c = torch.cat((p_cw1,p_cw2),dim=1)
        # p_w = torch.cat((p_cw1,p_cw2),dim=3)
        # # p = torch.cat((p))
        x1 = x1.view(B, C, H, W)
        x2 = x2.view(B, C, H, W)
        x3 = x3.view(B, C, H, W)
        x4 = x4.view(B, C, H, W)

    # img1 = x1
        # img2 = y
        ssim_value1 = self.ssim(x1,y)
        # ssim_value1 = self.ssim(x1.cpu().detach().numpy(), y.cpu().detach().numpy(),multichannel=True)
        ssim_value2 = self.ssim(x2, y)
        ssim_value3 = self.ssim(x3, y)
        ssim_value4 = self.ssim(x4, y)
        weights = torch.tensor([ssim_value1,ssim_value2,ssim_value3,ssim_value4])
        weights_softmax = F.softmax(weights,dim=0)
        weighted_image1 = x1 * weights_softmax[0]
        weighted_image2 = x2 * weights_softmax[1]
        weighted_image3 = x3 * weights_softmax[2]
        weighted_image4 = x4 * weights_softmax[3]
        x = weighted_image1 + weighted_image2 + weighted_image3 + weighted_image4

        return x
            # 应用1xk的卷积
# if __name__ == '__main__':
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     fourconv(7,128,128)
# torch.cuda.empty_cache()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# x1 = torch.randn(1, 3,512,512)
# x2 = torch.randn(1,3,512,512)
# a=fourconv(3,7,3,3)
# out=a(x1,x2)
# print(out.shape)

