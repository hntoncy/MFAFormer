from mynet.D2AFormer.D_Trans import *
from mynet.D2AFormer.D_CNN import *
class D2Block(nn.Module):
    def __init__(self,in_channels,out_channels,img_size,emb_size=96 ):
        super().__init__()
        self.trans = Trans(in_channels , emb_size=emb_size, img_size=img_size ,num_heads = 8,dropout=0,expansion=4)
        self.CNN = fourconv(unit=4,k=7,in_channels=in_channels,out_channels=out_channels,win_size=11, size_average=True, sigma=1.5)
    def forward(self, x,y,z):#x是CFE的输出(跳跃连接），y是上一层D2（解码器），z是MLFS的输出
        a = self.CNN(y,z)
        a = torch.sigmoid(a)
        b = a+x 
        b = self.trans(b)
        b = torch.sigmoid(b)
        # print("D2_b=",b.shape)
        # print("D2_x=",x.shape)
        out = b+y

        return out
