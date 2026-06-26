import numpy as np
import torch
from torchvision.ops import nms
from torchvision.ops import RoIAlign
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

#--------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
#--------------------------------------------#
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    return anchor.reshape((K * A, 4)).astype(np.float32)

#--------------------------------------------#
#   生成基础的先验框
#--------------------------------------------#
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i, ratio in enumerate(ratios):
        for j, scale in enumerate(anchor_scales):
            h = base_size * scale * np.sqrt(ratio)
            w = base_size * scale / np.sqrt(ratio)
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = -h / 2.
            anchor_base[index, 1] = -w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

#--------------------------------------------#
#   通过回归预测调整先验框
#--------------------------------------------#
def loc2bbox(src_bbox, loc):
    if src_bbox.size(0) == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx, dy, dw, dh = loc[:, 0::4], loc[:, 1::4], loc[:, 2::4], loc[:, 3::4]
    ctr_x = dx * src_width.unsqueeze(-1) + src_ctr_x.unsqueeze(-1)
    ctr_y = dy * src_height.unsqueeze(-1) + src_ctr_y.unsqueeze(-1)
    w = torch.exp(dw) * src_width.unsqueeze(-1)
    h = torch.exp(dh) * src_height.unsqueeze(-1)

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h
    return dst_bbox

#--------------------------------------------#
#   Proposal Layer: 建议框生成
#--------------------------------------------#
class ProposalCreator():
    def __init__(self, mode, nms_iou=0.7, n_train_pre_nms=12000, n_train_post_nms=600,
                 n_test_pre_nms=3000, n_test_post_nms=300, min_size=16):
        self.mode = mode
        self.nms_iou = nms_iou
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        n_pre_nms = self.n_train_pre_nms if self.mode == "training" else self.n_test_pre_nms
        n_post_nms = self.n_train_post_nms if self.mode == "training" else self.n_test_post_nms

        anchor = torch.from_numpy(anchor).to(loc.device)
        roi = loc2bbox(anchor, loc)

        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        min_size = self.min_size * scale
        keep = ((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size)
        roi = roi[keep]
        score = score[keep]

        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order]
        score = score[order]

        keep = nms(roi, score, self.nms_iou)[:n_post_nms]
        return roi[keep]

#--------------------------------------------#
#   RPN网络定义
#--------------------------------------------#
class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 feat_stride=16, mode="training"):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(mode)

        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.score.weight, 0, 0.01)
        nn.init.normal_(self.loc.weight, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        x = F.relu(self.conv1(x))

        rpn_locs = self.loc(x).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(x).permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        rpn_fg_scores = F.softmax(rpn_scores, dim=-1)[:, :, 1].contiguous().view(n, -1)
        anchor = _enumerate_shifted_anchor(self.anchor_base, self.feat_stride, h, w)

        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),), dtype=torch.int32, device=roi.device)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        # 只返回建议区域 (rois)
        return rois
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
class RegionConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, feat_stride=16, scale_factor=2):
        """
        初始化 RegionConvolution 类。

        Args:
            in_channels (int): 输入图像的通道数。
            out_channels (int): 卷积操作的输出通道数。
            feat_stride (int): 特征步长，默认为16。
            scale_factor (int): 特征尺度降低和恢复的因子，默认为2。
        """
        super(RegionConvolution, self).__init__()
        self.rpn = RegionProposalNetwork(in_channels=in_channels, mid_channels=out_channels, ratios=[0.5, 1, 2],
                                         anchor_scales=[8, 16, 32], feat_stride=feat_stride, mode="training")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)  # 下采样
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)  # 上采样
        # self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.qkv = nn.Linear(out_channels, out_channels * 3)

    def forward(self, x, img_size, scale=1.):
        """
        在建议区域内应用卷积操作，并降低特征尺度后再恢复。

        Args:
            x (torch.Tensor): 输入图像，形状为 (N, C, H, W)。
            img_size (tuple): 输入图像的尺寸，格式为 (height, width)。
            scale (float): 缩放因子，默认为1。

        Returns:
            torch.Tensor: 应用卷积后的输出。
        """
        # 通过 RPN 生成建议区域
        rois = self.rpn(x, img_size, scale)

        # 创建掩码：初始为全0
        n, c, h, w = x.shape
        mask = torch.zeros((h, w), dtype=torch.bool, device=x.device)

        # 填充掩码：为每个建议区域标记为1
        for roi in rois:
            x1, y1, x2, y2 = roi.int()
            mask[y1:y2, x1:x2] = True

        # 扩展掩码到与输入图像相同的形状
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(x)

        # 将掩码应用到图像，其他区域置为0
        masked_x = x * mask.float()

        # 下采样
        x_downsampled = self.downsample(masked_x)

        # 应用注意力机制
        batch_size, C, H, W = x_downsampled.size()
        # Q = self.query(x_downsampled).view(batch_size, C, -1)  # (B, C, N)
        # K = self.key(x_downsampled).view(batch_size, C, -1)  # (B, C, N)
        # V = self.value(x_downsampled).view(batch_size, C, -1)  # (B, C, N)

        # 调整维度顺序以适配线性层
        x_permuted = x_downsampled.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_flat = x_permuted.reshape(-1, C)  # (B*H*W, C)
        qkv = self.qkv(x_flat)  # (B*H*W, 3*C)
        qkv = qkv.view(batch_size, H, W, 3, C)  # (B, H, W, 3, C)

        # 分离Q, K, V并调整维度顺序
        Q = qkv[..., 0, :].permute(0, 3, 1, 2).view(batch_size,C,-1)  # (B, C, H, W)
        K = qkv[..., 1, :].permute(0, 3, 1, 2).view(batch_size,C,-1)
        V = qkv[..., 2, :].permute(0, 3, 1, 2).view(batch_size,C,-1)

        attn_scores = torch.bmm(Q.transpose(1, 2), K)  # (B, N, N)
        attn_scores = attn_scores / (C ** 0.5)  # 缩放
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N, N)

        out = torch.bmm(attn_weights, V.transpose(1, 2))  # (B, N, C)
        out = out.transpose(1, 2).view(batch_size, C, H, W)

        # 上采样恢复尺度
        out_upsampled = self.upsample(out)

        # 将处理后的特征图与原始输入特征图进行融合
        out_fused = out_upsampled * mask.float() + x * (~mask).float()

        return out_fused
# 定义 RegionConvolution 类
# class RegionConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, feat_stride=16):
#         """
#         初始化 RegionConvolution 类。
#
#         Args:
#             in_channels (int): 输入图像的通道数。
#             out_channels (int): 卷积操作的输出通道数。
#             rpn (nn.Module): RegionProposalNetwork 实例，用于生成建议区域。
#             feat_stride (int): 特征步长，默认为16。
#         """
#         super(RegionConvolution, self).__init__()
#         self.rpn = RegionProposalNetwork(in_channels=in_channels, mid_channels=out_channels, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
#                  feat_stride=16, mode="training")
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.feat_stride = feat_stride
#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#
#     def forward(self, x, img_size, scale=1.):
#         """
#         在建议区域内应用卷积操作。
#
#         Args:
#             x (torch.Tensor): 输入图像，形状为 (N, C, H, W)。
#             img_size (tuple): 输入图像的尺寸，格式为 (height, width)。
#             scale (float): 缩放因子，默认为1。
#
#         Returns:
#             torch.Tensor: 应用卷积后的输出。
#         """
#         # 通过 RPN 生成建议区域
#         res=x
#         rois = self.rpn(x, img_size, scale)
#
#         # 创建掩码：初始为全0
#         n, c, h, w = x.shape
#         mask = torch.zeros((h, w), dtype=torch.bool, device=x.device)
#
#         # 填充掩码：为每个建议区域标记为1
#         for roi in rois:
#             x1, y1, x2, y2 = roi.int()
#             mask[y1:y2, x1:x2] = 1
#
#         # 扩展掩码到与输入图像相同的形状
#         mask = mask.unsqueeze(0).unsqueeze(0).expand_as(x)
#
#         # 将掩码应用到图像，其他区域置为0
#         x = x * mask.float()
#         batch_size, C, H, W = x.size()
#         Q = self.query(x)  # (B, C, H, W)
#         K = self.key(x)  # (B, C, H, W)
#         V = self.value(x)  # (B, C, H, W)
#
#         # 将空间维度展平为序列
#         Q = Q.view(batch_size, C, -1)  # (B, C, N)
#         K = K.view(batch_size, C, -1)  # (B, C, N)
#         V = V.view(batch_size, C, -1)  # (B, C, N)
#
#         # 计算注意力权重
#         attn_scores = torch.bmm(Q.transpose(1, 2), K)  # (B, N, N)
#         attn_scores = attn_scores / (C ** 0.5)  # 缩放
#         attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N, N)
#
#         # 计算加权值
#         out = torch.bmm(attn_weights, V.transpose(1, 2))  # (B, N, C)
#         out = out.transpose(1, 2).view(batch_size, C, H, W)
#
#
#         # 在掩码后的图像上进行卷积操作
#         return out

# -------------------只需要将卷积改成自注意力就大功告成啦！！！--------------

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
#     rpn = RegionProposalNetwork()  # 请根据你的 RPN 配置进行初始化
#
#     # 创建 RegionConvolution 实例
#     region_conv = RegionConvolution(in_channels=3, out_channels=2)
#
#     # 输入图像和图像尺寸
#     img_size = image_tensor.shape[2:]  # 获取图像尺寸 (H, W)
#
#     # 通过 RegionConvolution 进行卷积操作
#     output = region_conv(image_tensor, img_size)
#
#     # 显示卷积输出的结果
#     print(output.shape)
#     output_image = output[0, 0, :, :].detach().cpu().numpy()
#     plt.imshow(output_image, cmap='gray')  # 使用灰度色图显示图像
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()
#
#
#
# x = torch.randn(1, 512, 64, 64)
#
# # 设置模型参数
# mode = 'training'  # 或者 'testing'，根据实际需要调整
# scale = 1.0  # 调整缩放比例
#
# # 创建RPN网络
# rpn = RegionConvolution(in_channels=512, out_channels=512, feat_stride=16)
# img_size=(64,64)
# 进行前向传播
# rois = rpn(x, img_size)
#
# 打印结果
# print(f"RPN Locations (rpn_locs) Shape: {rpn_locs.shape}")
# print(f"RPN Scores (rpn_scores) Shape: {rpn_scores.shape}")
# print(f"Proposals (rois) Shape: {rois}")
# print(f"ROI Indices (roi_indices) Shape: {roi_indices.shape}")
# print(f"Anchor Shape: {anchor.shape}")