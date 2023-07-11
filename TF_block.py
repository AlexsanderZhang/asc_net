"""
@ TF_block.py
"""
import torch
from torch import nn
from Cov_block import Mcp


# SE老版本
class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


#   测试线性层与卷积层参数
class test_layer(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        self.fc = nn.Linear(inc, ouc, bias=False)
        self.cn = nn.Conv2d(inc, ouc, 1, bias=False)

    def forward(self, x):
        y = self.cn(x)
        return y


#   像素混合编码层
class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size: int,
                 inc: int,
                 embed_dim: int,
                 drop_ratio: float = 0.5,
                 nor_method: str = 'GN'):
        super().__init__()
        # self.img_size = img_size
        self.patch_size = patch_size
        # grid_size = [img_size[0]//patch_size[0], img_size[1]//patch_size[1]]
        # self.num_patch = grid_size[0]*grid_size[1]

        self.proj = nn.Conv2d(inc, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        if nor_method == 'BN':
            self.norm = nn.BatchNorm2d(embed_dim)
        elif nor_method == 'GN':
            self.norm = nn.GroupNorm(int(embed_dim / 16), embed_dim)
        self.dropout = nn.Dropout(p=drop_ratio)

    def forward(self, x):
        # b, ch, h, w = x.shape
        x = self.norm(self.proj(x))
        # x = x.flatten(2)        # b, c, h, w -> b, c, h*w
        # x = x.transpose(1, 2)       # b, c, h*w -> b, h*w, c

        embed = self.dropout(x)
        return embed


class NewEmbed(nn.Module):
    def __init__(self, img_size, patch_size, inc, embed_dim, nor_method):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size[0])
        self.proj = nn.Conv2d(inc, embed_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        b, ch, h, w = x.shape
        x = self.unfold(x)      # b, c*k*k, l
        x = x.view(b, ch, self.patch_size[0], self.patch_size[1], -1)     # b, c, k, k, l
        x = x.permute(0, 4, 1, 2, 3)        # b, l, c, k, k
        x = x[0]        # l, c, k, k
        x = self.proj(x)
        x = x.flatten(2)
        x = x.flatten(0, 1)         # embed_dim, k*k
        x = x.transpose(0, 1)
        return x


# TF原版Multi_Heads_Self_Attention
class MHSA(nn.Module):
    def __init__(self, inc: int, groups: int, dropratio: float, ouc=None, qkv_bias=True):
        super(MHSA).__init__()
        self.al_num_hs = inc
        self.num_hs = groups
        self.dim_1h = self.al_num_hs//self.num_hs    # 每个头通道数
        self.out_dim = ouc if ouc is not None else inc

        self.to_query = nn.Linear(inc, self.al_num_hs, bias=qkv_bias)
        self.to_key = nn.Linear(inc, self.al_num_hs, bias=qkv_bias)
        self.to_value = nn.Linear(inc, self.al_num_hs, bias=qkv_bias)

        self.proj = nn.Linear(self.al_num_hs, self.out_dim)
        self.scale = self.dim_1h ** (-0.5)   # 缩放系数
        self.act = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=dropratio)

    def __split_head(self, temp):
        b, hw, c = temp.shape
        # 将c拆分为 b, wh, 检测头数， 每个头通道数
        # b, wh, g, d -> b, g, wh, d
        temp = temp.reshape(b, hw, self.num_hs, int(c//self.num_hs)).permute(0, 2, 1, 3)
        return temp

    def forward(self, embed_index):
        b, hw, c = embed_index.shape
        mul_q = self.to_query(embed_index)
        mul_k = self.to_key(embed_index)
        mul_v = self.to_value(embed_index)

        q = self.__split_head(mul_q)
        k = self.__split_head(mul_k).transpose(2, 3)     # 转置方便阵乘
        v = self.__split_head(mul_v)

        attn_score = torch.matmul(q, k) * self.scale     # b, g, wh, wh
        attn_score = self.act(attn_score)
        attn_score = self.drop(attn_score)
        # 阵乘结果为 b, g, wh, d
        dot = torch.matmul(attn_score, v).transpose(1, 2).reshape(b, hw, c)
        attn_out = self.proj(dot)
        attn_out = self.drop(attn_out)
        return attn_out


# 全卷积版Multi_Heads_Channels_Attention
class MHCA(nn.Module):
    def __init__(self,
                 inc: int,
                 ouc: int,
                 h1_dims: int = 32,
                 drop_ratio: float = 0.1,
                 qkv_bias: bool = True,
                 ):
        super(MHCA, self).__init__()
        self.al_num_hs = inc
        self.dim_1h = h1_dims    # 每个头通道数
        self.num_hs = self.al_num_hs//self.dim_1h
        self.out_dim = ouc if ouc is not None else inc

        self.qkv_conv = nn.Conv2d(self.al_num_hs, self.out_dim, 1, bias=qkv_bias)
        self.act = nn.Softmax(dim=-1)

        self.proj = nn.Conv2d(self.al_num_hs, self.out_dim, 1, bias=qkv_bias)
        self.drop = nn.Dropout(p=drop_ratio)

    def __split_head(self, x):
        x = x.flatten(2)
        b, c, hw = x.shape
        temp = x.reshape(b, self.num_hs, self.dim_1h, hw)   # b, g, d, hw
        return temp

    def forward(self, embedx):
        b, c, h, w = embedx.shape
        proj_q = self.qkv_conv(embedx)
        # DropKey   @ 2023CVPR: DropKey
        proj_k = self.drop(self.qkv_conv(embedx))
        proj_v = self.qkv_conv(embedx)
        #  b, c, h, w -> b, g, d, hw

        q = self.__split_head(proj_q)
        k = self.__split_head(proj_k).transpose(2, 3)   # b, g, hw, d
        v = self.__split_head(proj_v)

        scale = (h * w) ** (-0.5)  # 缩放系数
        att_score = torch.matmul(q, k) * scale     # b, g, d, d
        att_score = self.act(att_score)

        # att_score = self.drop(att_score)

        att_score = torch.matmul(att_score, v).reshape(b, c, h, w)

        att_score = self.proj(att_score)
        att_score = self.drop(att_score)

        return att_score


# Attention_Preceptron_Block
class APB(nn.Module):
    def __init__(self,
                 inc: int,
                 ouc: int,
                 head_dims: int,
                 embed_ratio: float = 1.,
                 drop_ratio: float = 0.1,
                 ):
        super(APB, self).__init__()
        self.embed_dim = int(inc * embed_ratio)
        self.embed = PatchEmbed(1, inc, self.embed_dim, drop_ratio)
        self.atte = MHCA(self.embed_dim, ouc, head_dims, drop_ratio)
        self.mcp = Mcp(ouc, ouc, drop_ratio=drop_ratio)

    def forward(self, x):
        # b, c, h, w = x.shape
        x = self.embed(x)
        att_x = self.atte(x)
        x = x + att_x
        mcp_x = self.mcp(x)
        out = x + mcp_x
        return out

