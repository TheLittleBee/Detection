import torch
import torch.nn as nn
from collections import OrderedDict
from ..layer import mobilenet_layer as m_layer
from ..layer import layer as l_layer
from ._backbone import Backbone
import numpy as np


class MobileNetv2(nn.Module, Backbone):
    """
    mobilenetv2
    """

    def __init__(self, width_mult=1):
        """ Network initialisation """
        super(MobileNetv2, self).__init__()

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
        bottleneck = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if i == 0:
                    features.append(m_layer.InvertedResidual(input_channel, output_channel, s, t))
                else:
                    features.append(m_layer.InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
            bottleneck.append(nn.Sequential(*features))
        # layer19 = nn.Sequential(
        #     nn.Conv2d(input_channel, 1280*width_mult,1,bias=False),
        #     nn.BatchNorm2d(1280*width_mult),
        #     nn.ReLU(inplace=True),
        # )
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
                ('Conv5', nn.Sequential(bottleneck[5], bottleneck[6])),
            ]),
            # OrderedDict([
            #     ('Layer19', layer19),
            # ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stage1 = self.layers[0](x)
        stage2 = self.layers[1](stage1)
        stage3 = self.layers[2](stage2)
        stage4 = self.layers[3](stage3)
        stage5 = self.layers[4](stage4)
        # stage5 = self.layers[5](stage5)

        return stage5, stage4, stage3
