from ZJW.mynet.MSSBlock.biformer import Block
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


class MSS_Merge_layer(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义三个卷积层
        self.conv1 = nn.Conv2d(128,128, kernel_size=15, padding=7)
        self.conv2 = nn.Conv2d(256,256, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(512, 512, 2, stride=2)

        # 定义深度可分离卷积
        self.depthwise = nn.Conv2d(896,896, kernel_size=3, padding=1,
                                   groups=896)
        self.pointwise = nn.Conv2d(896, 256, kernel_size=1)

    # 前向传播
    def forward(self, Y1, Y2, Y3):
        if Y1.size(1)>256:
            Y1=self.conv3(Y1)
            Y1 = self.up(Y1)
        elif Y1.size(1)<256:
            Y1=self.conv1(Y1)
            Y1=self.down(Y1)
        else:
            Y1=self.conv2(Y1)

        if Y2.size(1)>256:
            Y2=self.conv3(Y2)
            Y2 = self.up(Y2)
        elif Y2.size(1)<256:
            Y2=self.conv1(Y2)
            Y2=self.down(Y2)
        else:
            Y2=self.conv2(Y2)
        if Y3.size(1)>256:
            Y3=self.conv3(Y3)
            Y3 = self.up(Y3)
        elif Y3.size(1)<256:
            Y3=self.conv1(Y3)
            Y3=self.down(Y3)
        else:
            Y3=self.conv2(Y3)
        Y = torch.cat([Y1, Y2, Y3], dim=1)
        Y = self.depthwise(Y)
        Y = self.pointwise(Y)

        return Y


class MSS_stage2(nn.Module):
    def __init__(self, in_channels=128, out_channels=256, drop_path=0., layer_scale_init_value=-1,
                num_heads=8, n_win=4, qk_dim=None, qk_scale=None,
                kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                mlp_ratio=4, mlp_dwconv=False,
                side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=15, padding=7)

        self.ria = Block(dim=out_channels, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
                         num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale,
                         kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                         kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                         topk=topk, param_attention=param_attention, param_routing=param_routing,
                         diff_routing=diff_routing, soft_routing=soft_routing,
                         mlp_ratio=mlp_ratio, mlp_dwconv=mlp_dwconv,
                         side_dwconv=side_dwconv, before_attn_dwconv=before_attn_dwconv, pre_norm=pre_norm,
                         auto_pad=auto_pad)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        x1 = self.ria(x)
        out = self.relu(self.bn1(self.conv1(x1)))
        out = self.bn2(self.conv2(out))
        out = out + x  # 残差连接
        out = self.relu(out)
        return out
class MSS_stage3(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, drop_path=0., layer_scale_init_value=-1,
                num_heads=8, n_win=4, qk_dim=None, qk_scale=None,
                kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                mlp_ratio=4, mlp_dwconv=False,
                side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=7, padding=3)

        self.ria = Block(dim=out_channels, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
                         num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale,
                         kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                         kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                         topk=topk, param_attention=param_attention, param_routing=param_routing,
                         diff_routing=diff_routing, soft_routing=soft_routing,
                         mlp_ratio=mlp_ratio, mlp_dwconv=mlp_dwconv,
                         side_dwconv=side_dwconv, before_attn_dwconv=before_attn_dwconv, pre_norm=pre_norm,
                         auto_pad=auto_pad)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.ria(x)
        out = self.relu(self.bn1(self.conv1(x1)))
        out = self.bn2(self.conv2(out))
        out = out + x  # 残差连接
        out = self.relu(out)
        return out
class MSS_stage4(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, drop_path=0., layer_scale_init_value=-1,
                num_heads=8, n_win=4, qk_dim=None, qk_scale=None,
                kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                mlp_ratio=4, mlp_dwconv=False,
                side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels , kernel_size=3, padding=1)

        self.ria = Block(dim=out_channels, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
                         num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale,
                         kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                         kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                         topk=topk, param_attention=param_attention, param_routing=param_routing,
                         diff_routing=diff_routing, soft_routing=soft_routing,
                         mlp_ratio=mlp_ratio, mlp_dwconv=mlp_dwconv,
                         side_dwconv=side_dwconv, before_attn_dwconv=before_attn_dwconv, pre_norm=pre_norm,
                         auto_pad=auto_pad)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x1 = self.ria(x)
        out = self.relu(self.bn1(self.conv1(x1)))
        out = self.bn2(self.conv2(out))
        out = out + x  # 残差连接
        out = self.relu(out)
        return out

class Fusion(nn.Module):
    def __init__(self, dim, qkv_bias=False, in_channels=128,out_channels=128,qk_scale=4, attn_drop=0., proj_drop=0.,drop_path=0., layer_scale_init_value=-1,num_heads=8,
                  n_win=4, qk_dim=None,
                kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                mlp_ratio=4, mlp_dwconv=False,
                side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()



        self.scale = qk_scale ** -0.5

        self.wq = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.wk = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.wv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Merge_layer = MSS_Merge_layer()
        self.stage2 = MSS_stage2(128, 256, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
                num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale,
                kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio, kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention, param_routing=param_routing, diff_routing=diff_routing, soft_routing=soft_routing,
                mlp_ratio=mlp_ratio, mlp_dwconv=mlp_dwconv,
                side_dwconv=side_dwconv, before_attn_dwconv=before_attn_dwconv, pre_norm=pre_norm, auto_pad=auto_pad)
        self.stage3 = MSS_stage3(256, 256, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
                num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale,
                kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio, kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention, param_routing=param_routing, diff_routing=diff_routing, soft_routing=soft_routing,
                mlp_ratio=mlp_ratio, mlp_dwconv=mlp_dwconv,
                side_dwconv=side_dwconv, before_attn_dwconv=before_attn_dwconv, pre_norm=pre_norm, auto_pad=auto_pad)
        self.stage4 = MSS_stage4(512, 256, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value,
                num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale,
                kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio, kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention, param_routing=param_routing, diff_routing=diff_routing, soft_routing=soft_routing,
                mlp_ratio=mlp_ratio, mlp_dwconv=mlp_dwconv,
                side_dwconv=side_dwconv, before_attn_dwconv=before_attn_dwconv, pre_norm=pre_norm, auto_pad=auto_pad)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)


    def forward(self, x1,x2,x3):


        x4 = self.Merge_layer(x1, x2, x3)
        if x1.size(1) > 256:
            x1 = self.stage4(x1)
        elif x1.size(1) < 256:
            x1=self.stage2(x1)
        else:
            x1=self.stage3(x1)
        if x2.size(1) > 256:
            x2 = self.stage4(x2)
        elif x2.size(1) < 256:
            x2=self.stage2(x2)
        else:
            x2=self.stage3(x2)
        if x3.size(1) > 256:
            x3 = self.stage4(x3)
        elif x3.size(1) < 256:
            x3=self.stage2(x3)
        else:
            x3=self.stage3(x3)

        q = self.wq(x4) # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x2)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N


        attn += x1  # Add bias to attention scores
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = x.permute(0, 2, 3, 1)
        x = self.proj(self.norm2(x))
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        out = self.relu(self.bn1(self.conv1(x4)))
        out = self.bn2(self.conv2(out))
        out = out + x  # 残差连接
        out = self.relu(out)
        return out
# if __name__ == "__main__":
#     # 加载图像并转换为 PyTorch 张量
#     img_path = "D:\\PycharmProjects\\pythonProject1\\Unet\\data02\\test\\1_res.png"  # 将此路径替换为实际图像路径
#     image_pil = Image.open(img_path).convert("RGB")  # 转换为RGB图像
#
#     # 定义图像预处理
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),  # 调整图像大小
#         transforms.ToTensor(),        # 转换为张量
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
#     ])
#
#     image_tensor = transform(image_pil).unsqueeze(0)  # 转换为形状 (1, C, H, W)
#
#     # 假设初始化了 RegionProposalNetwork 类实例 rpn
#     # rpn = RegionProposalNetwork()  # 请根据你的 RPN 配置进行初始化
#
#     # 创建 RegionConvolution 实例
#     region_conv = Fusion(dim=3, qkv_bias=False, in_channels=128,out_channels=128,qk_scale=4, attn_drop=0., proj_drop=0.,drop_path=0., layer_scale_init_value=-1,num_heads=8,
#                   n_win=4, qk_dim=None,
#                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
#                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
#                 mlp_ratio=4, mlp_dwconv=False,
#                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False)
#
#     # 输入图像和图像尺寸
#     img_size = image_tensor.shape[2:]  # 获取图像尺寸 (H, W)
#
#     # 通过 RegionConvolution 进行卷积操作
#     output = region_conv(image_tensor, img_size)
#
#     # 显示卷积输出的结果
#     # print(output.shape)
#     # output_image = output[0, 0, :, :].detach().cpu().numpy()
#     # plt.imshow(output_image, cmap='gray')  # 使用灰度色图显示图像
#     # plt.axis('off')  # 关闭坐标轴
#     # plt.show()
# mss_moudal=Fusion(dim=256)
# a = MSS_Merge_layer(128, 256, 512, 128, 256,512)
# b=Fusion(256,512,256)
# x1 = torch.randn(1, 128,256,256)
# x2 = torch.randn(1, 256,128,128)
# x3 = torch.randn(1, 512, 64,64)
# # a=Fusion(dim=256)
# # out_put =a (x1,x2,x3)
# out_put=b(x3,x2,x1)
# print(out_put.shape)