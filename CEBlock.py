from ZJW.mynet.CE.CE_Trans import *
from ZJW.mynet.CE.CE_CNN import *
class CEBlock(nn.Module):
    def __init__(self,in_channels,out_channels,img_size,emb_size=96 ):
        super().__init__()
        self.trans = Trans(in_channels , emb_size=emb_size, img_size=img_size ,num_heads = 8,dropout=0,expansion=4)
        self.CNN = fourconv(unit=4,k=7,in_channels=in_channels,out_channels=out_channels,win_size=11, size_average=True, sigma=1.5)
    def forward(self, x,y,z):#x是MAC的输出(跳跃连接），y是上一层CE（解码器），z是MSS的输出
        a = self.CNN(y,z)
        a = torch.sigmoid(a)
        b = a+x #2017有改动/3d用的*
        b = self.trans(b)
        b = torch.sigmoid(b)
        # print("CE_b=",b.shape)
        # print("CE_x=",x.shape)
        out = b+y

        return out
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     # 加载图像并转换为 PyTorch 张量
#     img_path1 = "D:\\PycharmProjects\\pythonProject1\\Unet\\data02\\test\\.png"  # 将此路径替换为实际图像路径
#     image_pil1 = Image.open(img_path1).convert("RGB")  # 转换为RGB图像
#     img_path2 = "D:\\PycharmProjects\\pythonProject1\\Unet\\data02\\test\\2.png"
#     img_path3 = "D:\\PycharmProjects\\pythonProject1\\Unet\\data02\\test\\3.png"
#     image_pil2 = Image.open(img_path2).convert("RGB")
#     image_pil3 = Image.open(img_path3).convert("RGB")
#     # 定义图像预处理
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),  # 调整图像大小
#         transforms.ToTensor(),        # 转换为张量
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
#     ])
#
#     image_tensor1 = transform(image_pil1).unsqueeze(0)  # 转换为形状 (1, C, H, W)
#     image_tensor2 = transform(image_pil2).unsqueeze(0)
#     image_tensor3 = transform(image_pil3).unsqueeze(0)
#     # 假设初始化了 RegionProposalNetwork 类实例 rpn
#     # rpn = CEBlock()  # 请根据你的 RPN 配置进行初始化
#
#     # 创建 RegionConvolution 实例
#     region_conv = CEBlock(in_channels=3, out_channels=3,img_size=512)
#
#     # 输入图像和图像尺寸
#     img_size = image_tensor1.shape[2:]  # 获取图像尺寸 (H, W)
#     img_size = image_tensor2.shape[2:]
#     img_size = image_tensor3.shape[2:]
#     # 通过 RegionConvolution 进行卷积操作
#     output = region_conv(image_tensor1,image_tensor2,image_tensor3)
#
#     # 显示卷积输出的结果
#     print(output.shape)
#     output_image = output[0, 0, :, :].detach().cpu().numpy()
#     plt.imshow(output_image, cmap='gray')  # 使用灰度色图显示图像
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()
