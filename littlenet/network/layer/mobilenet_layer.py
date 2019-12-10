import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        self.feature = None

        if abs(expand_ratio - 1) < .01:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # expand
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # depthwise
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # project
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Conv2dDepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, BN=True):
        super(Conv2dDepthWise, self).__init__()

        # Parameters

        if isinstance(kernel_size, (list, tuple)):
            padding = [int(ii / 2) for ii in kernel_size]
        else:
            padding = int(kernel_size / 2)

        # Layer
        if BN:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1),
            )

    def forward(self, x):
        x = self.layers(x)
        return x
