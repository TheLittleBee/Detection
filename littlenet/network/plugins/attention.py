import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(),
            nn.Linear(in_channel // reduction, in_channel),
        )

    def forward(self, x):
        xc = F.avg_pool2d(x, (x.size(2), x.size(3)))
        xc = xc.view(xc.size(0), -1)
        xc = self.mlp(xc)
        xc = xc.sigmoid().unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * xc


class SpatialAttention(nn.Module):
    def __init__(self, in_channel, kernel_size, block=1, reduction=2, bn=True):
        super(SpatialAttention, self).__init__()
        pad = (kernel_size - 1) // 2
        layers = []
        if block == 1:
            layers.append(nn.Conv2d(in_channel, 1, kernel_size, 1, pad, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(1))
        else:
            layers.append(nn.Conv2d(in_channel, in_channel // reduction, kernel_size, 1, pad, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(in_channel // reduction))
            layers.append(nn.ReLU(inplace=True))
            in_channel = in_channel // reduction
            for i in range(1, block - 1):
                layers.append(nn.Conv2d(in_channel, in_channel, kernel_size, 1, pad, bias=not bn))
                if bn:
                    layers.append(nn.BatchNorm2d(in_channel))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channel, 1, kernel_size, 1, pad, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        xc = self.conv(x)
        xc = xc.sigmoid()
        # xc = F.hardtanh(xc)
        return x * xc, xc
