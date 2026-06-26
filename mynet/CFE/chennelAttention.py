import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBlock(nn.Module):
    def __init__(self, in_channels):
        super(CustomBlock, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)  # DWConv
        self.pwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)  # PWConv
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Agp (Adaptive Global Pooling)
        self.prelu = nn.PReLU(num_parameters=in_channels)  # PReLU
        self.pwconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)  # PWConv (second instance)

    def forward(self, x):
        shortcut = x  # 残差连接
        out = self.dwconv(x)
        out = self.pwconv1(out)
        pooled = self.global_pool(out)
        out = self.prelu(pooled)
        out = self.pwconv2(out)
        out = F.softmax(out,dim=1) * shortcut+shortcut
        return out

# 测试
# x = torch.randn(1, 64, 32, 32)  # (batch, C, H, W)
# model = CustomBlock(64)
# output = model(x)
# print(output.shape)  # 输出应与输入形状相同 (1, 64, 32, 32)
