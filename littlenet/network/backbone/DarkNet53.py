import torch
import torch.nn as nn
from collections import OrderedDict
from ..layer import darknet_layer as d_layer
from ._backbone import Backbone
import numpy as np


class DarkNet53(nn.Module, Backbone):

    def __init__(self):
        super(DarkNet53, self).__init__()

        input_channels = 32

        # Network
        layer_list = [

            OrderedDict([
                ('Conv1', nn.Sequential(d_layer.Conv2dBatchLeaky(3, input_channels, 3, 1, 1)
                                        , d_layer.Stage(input_channels, 1))),
            ]),
            OrderedDict([
                ('Conv2', d_layer.Stage(input_channels * (2 ** 1), 2)),
            ]),
            OrderedDict([
                ('Conv3', d_layer.Stage(input_channels * (2 ** 2), 8)),
            ]),

            OrderedDict([
                ('Conv4', d_layer.Stage(input_channels * (2 ** 3), 8)),
            ]),

            OrderedDict([
                ('Conv5', d_layer.Stage(input_channels * (2 ** 4), 4)),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stage1 = self.layers[0](x)
        stage2 = self.layers[1](stage1)
        stage3 = self.layers[2](stage2)
        stage4 = self.layers[3](stage3)
        stage5 = self.layers[4](stage4)

        return stage5, stage4, stage3
