from mynet.D2AFormer.D_Trans import *
from mynet.CE.D_CNN import *
class D2Block(nn.Module):
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
        # print("D2_b=",b.shape)
        # print("D2_x=",x.shape)
        out = b+y

        return out
