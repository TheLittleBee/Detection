import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel, filter, out_channel, rate):
        super(ASPP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channel, filter, 1, bias=False),
            nn.BatchNorm2d(filter),
            nn.ReLU(inplace=True)
        ))
        for r in rate:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channel, filter, 3, 1, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(filter),
                nn.ReLU(inplace=True)
            ))
        self.conv = nn.Sequential(
            nn.Conv2d(filter * (len(rate) + 1), out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        layers = [l(x) for l in self.layers]
        x = self.conv(torch.cat(layers, dim=1))

        return x


class KSASPP(nn.Module):
    def __init__(self, in_channel, filter, out_channel, rate):
        super(KSASPP, self).__init__()
        self.point = nn.Sequential(nn.Conv2d(in_channel, filter, 1, bias=False),
                                   nn.BatchNorm2d(filter),
                                   nn.ReLU(inplace=True))
        self.conv_param = nn.Parameter(torch.zeros(filter, in_channel, 3, 3))
        nn.init.kaiming_normal_(self.conv_param, a=0.1, mode='fan_in')
        self.bn_relu = nn.ModuleList()
        for r in rate:
            self.bn_relu.append(nn.Sequential(nn.BatchNorm2d(filter),
                                              nn.ReLU(inplace=True)))
        self.project = nn.Sequential(
            nn.Conv2d(filter * (len(rate) + 1), out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.rate = rate

    def forward(self, x):
        layers = [self.point(x)]
        for i, r in enumerate(self.rate):
            xc = F.conv2d(x, self.conv_param, padding=r, dilation=r)
            layers.append(self.bn_relu[i](xc))
        x = self.project(torch.cat(layers, dim=1))

        return x
