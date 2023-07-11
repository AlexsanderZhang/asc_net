"""
@ filename: Cov_block.py

"""
import torch
from torch import nn


#   Multilayer_Conv_Preceptron
class Mcp(nn.Module):
    def __init__(self, inc: int,
                 ouc: int = 0,
                 mcp_ratio: float = 0.0625,
                 drop_ratio: float = 0.5,
                 is_ouc_diff: bool = False,
                 mcp_bias=True):
        super(Mcp, self).__init__()
        if is_ouc_diff is False:
            self.ouc = inc
        else:
            if ouc == 0:
                self.ouc = int(inc*mcp_ratio)
            else:
                self.ouc = ouc
        self.midc = int(inc*mcp_ratio)
        self.conv1 = nn.Conv2d(inc, self.midc, kernel_size=1, bias=mcp_bias)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(self.midc, self.ouc, kernel_size=1, bias=mcp_bias)
        self.drop = nn.Dropout(p=drop_ratio)

    def forward(self, x):
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.conv2(x))
        return x


#   Channel_Attention_block模块
class CABlock(nn.Module):
    def __init__(self, in_channel, drop_ratio):
        super(CABlock, self).__init__()
        self.avg_squeeze = nn.AdaptiveAvgPool2d(1)
        self.max_squeeze = nn.AdaptiveMaxPool2d(1)

        self.mcp = Mcp(in_channel, mcp_ratio=4, drop_ratio=drop_ratio)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.mcp(self.avg_squeeze(x))
        y2 = self.mcp(self.max_squeeze(x))
        y = self.act(y1 + y2)
        cab = x * y
        return cab


#   Atrous_Conv
class AConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(AConv, self).__init__()
        self.sppconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(int(out_channels / 16), out_channels),  # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sppconv(x)


#   Channel_Fusion_Block
class CFB(nn.Sequential):
    def __init__(self, in_channels, out_channels: int = 0, drop_ratio: float = 0.2, is_interp: bool = False):
        super(CFB, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.mcp = Mcp(in_channels, mcp_ratio=3, drop_ratio=drop_ratio)
        self.sign = is_interp

    def forward(self, x):
        size = x.shape[-2:]
        x = self.mcp(self.pooling(x))
        # inter_x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


#   Res_Spatial_Channel_Combined_Block
class SCCB(nn.Module):
    def __init__(self, in_channels, atrous_rates, drop_ratio, aspm_ratio: float = 0.25):
        super(SCCB, self).__init__()
        mid_channels = int(in_channels * aspm_ratio)
        out_channels = in_channels
        # 压缩通道，特征融合
        self.ch_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(int(mid_channels/16), mid_channels),  # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        modules = []
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(AConv(mid_channels, out_channels, rate1))
        modules.append(AConv(mid_channels, out_channels, rate2))
        modules.append(AConv(mid_channels, out_channels, rate3))
        self.convs = nn.ModuleList(modules)

        self.pool = CFB(in_channels, out_channels, drop_ratio=drop_ratio)
        self.act = nn.Sigmoid()

        self.project = nn.Sequential(
            nn.Conv2d(int(4 * out_channels), out_channels, 1, bias=False),
            nn.GroupNorm(int(out_channels/16), out_channels),  # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_ratio))

    def forward(self, x):
        sq_x = self.ch_conv(x)
        res = []
        for conv in self.convs:
            res.append(conv(sq_x))

        sq_x = torch.cat(res, dim=1)
        sq_x = torch.cat([sq_x, x], dim=1)

        ca_y = self.act(self.pool(x))

        res = self.project(sq_x) * ca_y
        res = x + res
        return res



