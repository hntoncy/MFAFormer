import torch
import torch.nn.functional as F

class GlobalSSIM(torch.nn.Module):
    def __init__(self, win_size=11, size_average=True, sigma=1.5,in_channels=1):
        super(GlobalSSIM, self).__init__()
        self.win_size = win_size
        self.size_average = size_average
        self.sigma = sigma
        self.channel = in_channels
        self.gaussian_window = self._create_gaussian_window()
    def _create_gaussian_window(self):
        gauss = torch.arange(self.win_size, dtype=torch.float).view(1, 1, 1, -1)
        gauss -= self.win_size // 2
        gauss = torch.exp(-(gauss ** 2) / (2 * self.sigma ** 2))
        gauss /= gauss.sum()
        return gauss.repeat(self.channel, 1, 1, 1)

    def _ssim(self, img1, img2):
        if img1.device != self.gaussian_window.device:
            self.gaussian_window = self.gaussian_window.to(img1.device)
        mu1 = F.conv2d(img1, self.gaussian_window, padding=self.win_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.gaussian_window, padding=self.win_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.gaussian_window, padding=self.win_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.gaussian_window, padding=self.win_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.gaussian_window, padding=self.win_size // 2, groups=self.channel) - mu1_mu2

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if self.size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        return ret, cs

    def forward(self, img1, img2):
        return self._ssim(img1, img2)[0]

# 使用示例
img1 = torch.randn(1, 3, 256, 256)  # 模拟输入图像1
img2 = torch.randn(1, 3, 256, 256) # 模拟输入图像2

# 实例化GlobalSSIM类
global_ssim = GlobalSSIM(in_channels=3).cuda()

# 计算全局SSIM
ssim_value = global_ssim(img1, img2).cuda()
print(f"Global SSIM similarity: {ssim_value.item()}")