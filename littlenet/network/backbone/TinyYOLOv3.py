import torch.nn as nn
from collections import OrderedDict

from ..layer import darknet_layer as d_layer
from ._backbone import Backbone


class TinyYOLOv3(nn.Module, Backbone):
    custom_layers = ()

    def __init__(self):
        super(TinyYOLOv3, self).__init__()

        layer_list = [
            OrderedDict([
                ('Conv1', nn.Sequential(
                    d_layer.Conv2dBatchLeaky(3, 16, 3, 1),
                    nn.MaxPool2d(2, 2),
                    d_layer.Conv2dBatchLeaky(16, 32, 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv2', nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    d_layer.Conv2dBatchLeaky(32, 64, 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv3', nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    d_layer.Conv2dBatchLeaky(64, 128, 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv4', nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    d_layer.Conv2dBatchLeaky(128, 256, 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv5', nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    d_layer.Conv2dBatchLeaky(256, 512, 3, 1),
                    nn.MaxPool2d(3, 1, 1),
                    d_layer.Conv2dBatchLeaky(512, 1024, 3, 1),
                    d_layer.Conv2dBatchLeaky(1024, 256, 1, 1),
                )),
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
