import torch
from torch import nn


class FeatureMapShifter(nn.Module):
    def __init__(self, window_size):
        super(FeatureMapShifter, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        shifted_map = torch.zeros_like(x, device=x.device)
        occupied = torch.zeros((H, W), dtype=torch.bool, device=x.device)
        displaced_pixels = []

        for i in range(H):
            for j in range(W):
                value = x[0, :, i, j]

                # 根据像素位置计算初始平移方向
                if i < H // 2 and j < W // 2:
                    di, dj = self.window_size, self.window_size  # 向右下平移
                elif i < H // 2 and j >= W // 2:
                    di, dj = self.window_size, -self.window_size  # 向左下平移
                elif i >= H // 2 and j < W // 2:
                    di, dj = -self.window_size, self.window_size  # 向右上平移
                else:
                    di, dj = -self.window_size, -self.window_size  # 向左上平移

                # 持续移动，直到找到空闲位置或超出边界
                i_new, j_new = i + di, j + dj
                while 0 <= i_new < H and 0 <= j_new < W and occupied[i_new, j_new]:
                    i_new += di
                    j_new += dj

                # 如果找到合法且未占用的位置
                if 0 <= i_new < H and 0 <= j_new < W:
                    shifted_map[0, :, i_new, j_new] = value
                    occupied[i_new, j_new] = True
                else:
                    # 超出边界，记录为需要随机放置的像素
                    displaced_pixels.append(value)

        # 随机放置超出边界的像素
        if displaced_pixels:
            free_positions = torch.nonzero(~occupied)
            for value in displaced_pixels:
                if free_positions.numel() > 0:
                    idx = torch.randint(0, free_positions.size(0), (1,)).item()
                    i_new, j_new = free_positions[idx]
                    shifted_map[0, :, i_new, j_new] = value
                    occupied[i_new, j_new] = True

        return shifted_map


# 示例使用
if __name__ == "__main__":
    # 假设输入特征图X的形状为 (1, C, H, W)，例如 C=3, H=512, W=512
    C, H, W = 3, 32, 32
    X = torch.randn(1, C, H, W)  # 用随机值填充特征图X

    # 设置平移窗口的大小
    window_size = 4  # 假设平移窗口的大小为1

    # 创建FeatureMapShifter对象
    shifter = FeatureMapShifter(window_size)

    # 获取平移后的特征图
    Y = shifter(X)

    # 打印原始特征图和移动后的特征图
    print("原特征图X:")
    print(X)
    print("\n移动后的特征图Y:")
    print(Y)
