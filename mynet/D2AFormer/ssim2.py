import torch
import torch.nn.functional as F


class GlobalSSIM(torch.nn.Module):
    def __init__(self, win_size=11, size_average=True, sigma=1.5, in_channels=1):
        super(GlobalSSIM, self).__init__()
        self.win_size = win_size
        self.size_average = size_average
        self.sigma = sigma
        self.channel = in_channels
        self.register_buffer("gaussian_window", self._create_gaussian_window())

    def _create_gaussian_window(self):
        gauss = torch.arange(self.win_size, dtype=torch.float).view(1, 1, 1, -1)
        gauss -= self.win_size // 2
        gauss = torch.exp(-(gauss ** 2) / (2 * self.sigma ** 2))
        gauss /= gauss.sum()
        return gauss.repeat(self.channel, 1, 1, 1)

    def _ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.gaussian_window, padding=self.win_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.gaussian_window, padding=self.win_size // 2, groups=self.channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.gaussian_window, padding=self.win_size // 2,
                             groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.gaussian_window, padding=self.win_size // 2,
                             groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.gaussian_window, padding=self.win_size // 2, groups=self.channel) - mu1_mu2

        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        v1, v2 = 2.0 * sigma12 + C2, sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        return ssim_map.mean() if self.size_average else ssim_map.mean(dim=(1, 2, 3))

    def forward(self, img1, img2):
        return self._ssim(img1, img2)
img1 = torch.randn(1, 3, 256, 256, dtype=torch.float32)  # 模拟输入图像1
img2 = torch.randn(1, 3, 256, 256, dtype=torch.float32)  # 模拟输入图像2

# 实例化GlobalSSIM类
global_ssim = GlobalSSIM(in_channels=3)

# 计算全局SSIM
ssim_value = global_ssim(img1, img2)
print(f"Global SSIM similarity: {ssim_value.item()}")
