from ZJW.mypart import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512,img_size=64 )
        self.up2 = Up(512, 256,img_size=128)
        self.up3 = Up(256, 128,img_size=256)
        self.up4 = convup(192, 64)
        # self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) #[1, 512, 64, 64]
        x5 = self.down4(x4) #[1, 1024, 32, 32]
        x6 = self.up1(x5,x4,x2,x3, x4)
        x7 = self.up2(x6, x3,x4,x2,x3)
        x8 = self.up3(x7,x2,x3,x4, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits
# x1 =torch.randn(1, 3, 512, 512)
# net = UNet(3,2)
# a= net(x1)
# print(a.shape)