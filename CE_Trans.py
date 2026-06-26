import torch
import torch.nn as nn
# from ViT import *
from ZJW.mynet.CE.CE_ViT import *
class AveragePooling1xW(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # x的形状为[B, C, H, W]，动态获取W
        B, C, H, W = x.shape
        # 应用平均池化，核大小为[1, W]，步长为[1, W]，padding为[0, 0]
        Avg_h = nn.functional.avg_pool2d(x, kernel_size=(1, W), stride=(1, W), padding=(0, 0)) #[28, 1]
        Max_h = nn.functional.max_pool2d(x, kernel_size=(1, W), stride=(1, W), padding=(0, 0))
        Avg_w = nn.functional.avg_pool2d(x, kernel_size=(H, 1), stride=(H, 1), padding=(0, 0)) #[1, 28]
        Max_w = nn.functional.max_pool2d(x, kernel_size=(H, 1), stride=(H, 1), padding=(0, 0))
        weights_h = Avg_h+Max_h #[28, 1]
        weights_w = Avg_w+Max_w #[1, 28]
        x_h = x*weights_h
        x_w = x*weights_w
        return x_h,x_w

class Trans(nn.Module):
    def __init__(self,in_channels, patch_size=32 , emb_size=768, img_size=224 ,num_heads = 8,dropout=0,expansion=4):
        super().__init__()
        self.pool = AveragePooling1xW()
        self.vit = Transblock(in_channels=in_channels,out_channels=in_channels, patch_size=patch_size , emb_size=emb_size, img_size=img_size ,num_heads = num_heads,dropout=dropout,expansion=expansion)
    def forward(self, x):
        x1,x2 = self.pool(x)
        x = self.vit(x1,x2)
        return x


#---------------------------------------------------------

model = Trans(in_channels=64, patch_size=16, emb_size=768,img_size=512 ,num_heads = 8,dropout=0,expansion=4)
x1 = torch.randn(1,64,512,512)
x2 = torch.randn(1,64,512,512)
output_image = model(x1)
# print("Input Image Shape:", input_image.type())
print("Output Image Shape:", output_image.shape)