import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ._backbone import Backbone

__all__ = ['vgg16', 'vgg16bn']

def vgg16():
    return VGG()

def vgg16bn():
    return VGG(True)

class VGG(nn.Module, Backbone):
    def __init__(self, bn=False):
        super(VGG, self).__init__()
        conv1 = self._make_layer(3, 64, 2, BN=bn)
        conv2 = self._make_layer(64, 128, 2, BN=bn)
        conv3 = self._make_layer(128, 256, 3, BN=bn, Ceil=True)
        conv4 = self._make_layer(256, 512, 3, BN=bn, Pool=False)
        conv5 = self._make_layer(512, 512, 3, BN=bn, Pool=False)

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
                ('MaxPool4', nn.MaxPool2d(2,2)),
                ('Conv5', conv5),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def _make_layer(self, inp, oup, blocks, BN=False, Pool=True, Ceil=False):
        input_channel = inp
        layer = []
        for i in range(blocks):
            conv = nn.Conv2d(input_channel, oup, 3, padding=1)
            if BN:
                layer += [conv, nn.BatchNorm2d(oup), nn.ReLU(inplace=True)]
            else:
                layer += [conv, nn.ReLU(inplace=True)]
            input_channel = oup
        if Pool:
            layer.append(nn.MaxPool2d(2, stride=2, ceil_mode=Ceil))
        return nn.Sequential(*layer)

    def forward(self, x):
        stage1 = self.layers[0](x)
        stage2 = self.layers[1](stage1)
        stage3 = self.layers[2](stage2)
        stage4 = self.layers[3](stage3)
        stage5 = self.layers[4](stage4)

        return stage5, stage4, stage3
