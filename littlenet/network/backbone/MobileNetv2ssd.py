import torch
import torch.nn as nn
from collections import OrderedDict
from ..layer import layer as l_layer
from ._backbone import Backbone
import numpy as np


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

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
            self.expand = nn.Sequential(
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
            )
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # project
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x, ex=False):
        xc = x
        expand = None
        if hasattr(self, 'expand'):
            xc = self.expand(x)
            expand = xc
        if self.use_res_connect:
            if ex: return x + self.conv(xc), expand
            return x+self.conv(xc)
        else:
            if ex: return self.conv(xc), expand
            return self.conv(xc)


class MobileNetv2SSD(nn.Module, Backbone):
    """
    mobilenetv2
    """

    def __init__(self, width_mult=1):
        """ Network initialisation """
        super(MobileNetv2SSD, self).__init__()

        # setting of inverted residual blocks
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)

        layer1 = l_layer.Conv2dBatchReLU(3, input_channel, 3, 2)
        layer15 = None
        bottleneck = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if c == 160 and i == 0:
                    layer15 = InvertedResidual(input_channel, output_channel, s, t)
                    input_channel = output_channel
                    continue
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
            bottleneck.append(nn.Sequential(*features))
        layer19 = nn.Sequential(
            nn.Conv2d(input_channel, 1280 * width_mult, 1, bias=False),
            nn.BatchNorm2d(1280 * width_mult),
            nn.ReLU(inplace=True),
        )
        layer_list = [
            OrderedDict([
                ('Conv1', nn.Sequential(layer1, bottleneck[0])),
            ]),
            OrderedDict([
                ('Conv2', bottleneck[1]),
            ]),
            OrderedDict([
                ('Conv3', bottleneck[2]),
            ]),
            OrderedDict([
                ('Conv4', nn.Sequential(bottleneck[3], bottleneck[4])),
            ]),
            OrderedDict([
                ('Layer15', layer15),
            ]),
            OrderedDict([
                ('Conv5', nn.Sequential(bottleneck[5], bottleneck[6])),
            ]),
            OrderedDict([
                ('Layer19', layer19),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stage1 = self.layers[0](x)
        stage2 = self.layers[1](stage1)
        stage3 = self.layers[2](stage2)
        conv4 = self.layers[3](stage3)
        layer15, stage4 = self.layers[4](conv4, True)
        conv5 = self.layers[5](layer15)
        stage5 = self.layers[6](conv5)

        return stage5, stage4, stage3
