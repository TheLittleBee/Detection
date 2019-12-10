import torch.nn as nn
from collections import OrderedDict

from ..layer import darknet_layer as d_layer
from ._backbone import Backbone


class DarkNet19(nn.Module, Backbone):
    def __init__(self):
        super(DarkNet19, self).__init__()
        conv1 = nn.Sequential(
            d_layer.Conv2dBatchLeaky(3, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            d_layer.Conv2dBatchLeaky(32, 64, 3, 1),
        )
        conv2 = self._make_layer(64, 128, 1)
        conv3 = self._make_layer(128, 256, 1)
        conv4 = self._make_layer(256, 512, 2)
        conv5 = self._make_layer(512, 1024, 2)

        layer_list = [
            OrderedDict([
                ('Conv1', conv1),
            ]),
            OrderedDict([
                ('Conv2', conv2),
            ]),
            OrderedDict([
                ('Conv3', conv3),
            ]),
            OrderedDict([
                ('Conv4', conv4),
            ]),
            OrderedDict([
                ('Conv5', conv5),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def _make_layer(self, inp, oup, blocks):
        layer = [nn.MaxPool2d(2, stride=2),
                 d_layer.Conv2dBatchLeaky(inp, oup, 3, 1)]
        for i in range(blocks):
            layer += [d_layer.Conv2dBatchLeaky(oup, inp, 1, 1),
                      d_layer.Conv2dBatchLeaky(inp, oup, 3, 1)]
        return nn.Sequential(*layer)

    def forward(self, x):
        stage1 = self.layers[0](x)
        stage2 = self.layers[1](stage1)
        stage3 = self.layers[2](stage2)
        stage4 = self.layers[3](stage3)
        stage5 = self.layers[4](stage4)

        return stage5, stage4, stage3
