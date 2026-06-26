import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from torchsummary import summary
import torch.utils.checkpoint as checkpoint

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels = 3, patch_size = 16, emb_size = 96, img_size = 224):
#         self.patch_size = patch_size
#         super().__init__()
#         self.projection = nn.Sequential(
#             # 使用一个卷积层而不是一个线性层 -> 性能增加
#             nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
#         # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
#         self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 , emb_size))
#
#     def forward(self, x: Tensor) -> Tensor:
#         b, _, _, _ = x.shape
#         x = self.projection(x)
#         # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
#         # 将cls token在维度1扩展到输入上
#         # x = torch.cat([cls_tokens, x], dim=1)
#         # 添加位置编码
#         # print("x:",x.shape, "position:",self.positions.shape)
#         x += self.positions
#         return x
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=96):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)  # B, E, H', W'
        x = self.rearrange(x)  # B, N, E
        return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size= 768, num_heads = 8, dropout = 0):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         # 使用单个矩阵一次性计算出queries,keys,values
#         self.qkv = nn.Linear(emb_size, emb_size * 3)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)
#         self.norm =  nn.LayerNorm(emb_size)
#
#     def forward(self, x1,x2,mask: Tensor = None) :
#         # 将queries，keys和values划分为num_heads
#         # print("1qkv's shape: ", self.qkv(x).shape)  # 使用单个矩阵一次性计算出queries,keys,values
#         x1 = self.norm(x1)
#         x2 = self.norm(x2)
#         res1 = x1
#         res2 = x2
#         qkv_1 = rearrange(self.qkv(x1), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
#         qkv_2 = rearrange(self.qkv(x2), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
#         # print("2qkv's shape: ", qkv.shape)
#
#         queries1, keys1, values1 = qkv_1[0], qkv_1[1], qkv_1[2]
#         queries2, keys2, values2 = qkv_2[0], qkv_2[1], qkv_2[2]
#         # print("queries's shape: ", queries.shape)
#         # print("keys's shape: ", keys.shape)
#         # print("values's shape: ", values.shape)
#
#         # 在最后一个维度上相加
#         energy_1 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys1)  # batch, num_heads, query_len, key_len
#         energy_2 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys2)
#         # print("energy's shape: ", energy.shape)
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy_1.mask_fill(~mask, fill_value)
#
#         scaling = self.emb_size ** (1 / 2)
#         # print("scaling: ", scaling)
#         att_1 = F.softmax(energy_1, dim=-1) / scaling
#         att_2 = F.softmax(energy_2, dim=-1) / scaling
#         # print("att1' shape: ", att.shape)
#         att_1 = self.att_drop(att_1)
#         att_2 = self.att_drop(att_2)
#         # print("att2' shape: ", att.shape)
#
#         # 在第三个维度上相加
#         out_1 = torch.einsum('bhal, bhlv -> bhav ', att_1, values2)
#         out_2 = torch.einsum('bhal, bhlv -> bhav ', att_2, values1)
# #-----------------------------融合机制-----------------------------
#         # print("out1's shape: ", out.shape)
#         out1 = rearrange(out_1, "b h n d -> b n (h d)")
#         out2 = rearrange(out_2, "b h n d -> b n (h d)")
#         out_1 = out1+res1
#         out_2 = out2+res2
#         out = out_1 + out_2
#         # print("out2's shape: ", out.shape)
#         out = self.projection(out)
#         # print("out3's shape: ", out.shape)
#         return out

# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x1, x2, mask: Tensor=None):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        qkv_1 = rearrange(self.qkv(x1), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        qkv_2 = rearrange(self.qkv(x2), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)

        queries1, keys1, values1 = qkv_1
        queries2, keys2, values2 = qkv_2

        energy_1 = torch.einsum('bhqd, bhkd -> bhqk', queries1, keys2)
        energy_2 = torch.einsum('bhqd, bhkd -> bhqk', queries2, keys1)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy_1.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att_1 = F.softmax(energy_1, dim=-1) / scaling
        att_2 = F.softmax(energy_2, dim=-1) / scaling

        att_1 = self.att_drop(att_1)
        att_2 = self.att_drop(att_2)

        out_1 = torch.einsum('bhal, bhlv -> bhav', att_1, values2)
        out_2 = torch.einsum('bhal, bhlv -> bhav', att_2, values1)

        out_1 = rearrange(out_1, "b h n d -> b n (h d)")
        out_2 = rearrange(out_2, "b h n d -> b n (h d)")
        out = out_1 + out_2

        return self.projection(out)
class MLPBlock(nn.Sequential):
    def __init__(self, emb_size, expansion= 4, dropout= 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )
class Transblock(nn.Module):
    def __init__(self,in_channels, out_channels,patch_size=1 , norm_layer=nn.LayerNorm,emb_size=96, img_size=224 ,num_heads = 8,dropout=0,expansion=4):
        super().__init__()
        # print(f"Received parameters: in_channels={in_channels}, patch_size={patch_size}, ...")
        # self.emb=patch_size**2*in_channels
        self.im_size = to_2tuple(img_size)
        self.pat_size = to_2tuple(patch_size)
        self.embed_size = emb_size
        self.num_layers = 1
        self.num_features = int(emb_size)
        self.norm = norm_layer(self.num_features)
        patches_resolution = [self.im_size[0] // self.pat_size[0], self.im_size[1] // self.pat_size[1]]
        # self.patches_resolution = patches_resolution
        self.embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_size=emb_size)
        self.att = MultiHeadAttention(emb_size=emb_size,num_heads=num_heads, dropout=dropout)
        self.down = PatchMerging(input_resolution=(patches_resolution[0] ,
                                            patches_resolution[1] ), dim=emb_size , norm_layer=norm_layer)
        self.mlp = MLPBlock(emb_size=emb_size,expansion=expansion,dropout=dropout)
        self.norm = nn.LayerNorm(normalized_shape=emb_size)
        self.drop = nn.Dropout(dropout)
        self.vit_img = img_size//patch_size
        self.patch_dim = patch_size
        # self.patch_wh=patch_size

        self.conv = torch.nn.Conv2d(int(emb_size), out_channels, kernel_size=3, stride=1, padding=1)
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)
    def forward(self, x1, x2):
        x1=self.embed(x1)
        x2=self.embed(x2)
        # print("x1=", x1.shape)
        x = self.att(x1,x2)
        res = x
        x = self.norm(x)
        x = self.mlp(x)
        x = self.drop(x)
        x = x+res
        x = self.down(x)
        # print(x.shape)

        x = self.norm(x)  # B L C

        x = x.permute(0, 3, 1, 2)
        # print("x=", x.shape)


        # x = rearrange(x, "b (x y) (c d) -> b c (x y d)",x=self.vit_img,y=self.vit_img,d=self.patch_c)
        # x = rearrange(x, "b c (x y d) -> b c x y ")
        # x = rearrange(x, ' b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
        #                         patch_x=self.patch_dim, patch_y=self.patch_dim,x=self.vit_img,y=self.vit_img)# 将输入图像重塑为patches序列
        return self.up(self.conv(x))
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim,dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # 还原到 (B, H, W, C)
        # print("norm",x.shape)

        return x


# 测试 Transblock


# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self,
#                  emb_size,
#                  drop_p = 0.,
#                  forward_expansion: int = 4,
#                  forward_drop_p: float = 0.,
#                  ** kwargs):
#         super().__init__(
#             ResidualAdd(nn.Sequential(
#                 MultiHeadAttention(emb_size, **kwargs),
#                 nn.Dropout(drop_p)
#             )),
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 MLPBlock(
#                     emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
#                 nn.Dropout(drop_p)
#             )
#             ))
#---------------------------后续恢复原图尺寸时查看书签中的transunet的最后整合阶段的rearrange(原文用的双线性插值上采样）-------------------------
# x2 = torch.randn(1,512,64,64)
# x1 = torch.randn(1,512,64,64)
# net=Transblock(in_channels = 512, out_channels=512,patch_size = 4, emb_size = 768, img_size = 64,num_heads = 8,dropout=0,expansion=4)
# # net = TransformerEncoderBlock(768)
# # B = embed(x1)
# A = net(x1,x2)
# print("transformer:",A.shape)
#
# embed = PatchEmbedding(in_channels = 3, patch_size = 16, emb_size = 768, img_size = 512)