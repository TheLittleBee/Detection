import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..layer.mobilenet_layer import Conv2dDepthWise
from ..layer.layer import Conv2dBatchReLU


class HeadBody(nn.Module):

    def __init__(self, nchannels, half_nchannels):
        super(HeadBody, self).__init__()
        in_nchannels = 2 * half_nchannels
        layers = [
            Conv2dDepthWise(nchannels, half_nchannels, 3, 1),
            Conv2dBatchReLU(half_nchannels, in_nchannels, 1, 1),
            Conv2dDepthWise(in_nchannels, half_nchannels, 3, 1),
        ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x


class Transition(nn.Module):

    def __init__(self, nchannels):
        super(Transition, self).__init__()
        half_nchannels = int(nchannels / 2)
        layers = [
            Conv2dBatchReLU(nchannels, half_nchannels, 1, 1),
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return F.interpolate(x, scale_factor=2)


class Head(nn.Module):

    def __init__(self, nchannels, nanchors, nclasses):
        super(Head, self).__init__()
        mid_nchannels = 2 * nchannels
        layer_list = [
            Conv2dBatchReLU(nchannels, mid_nchannels, 1, 1),
            Conv2dDepthWise(mid_nchannels, nchannels, 3, 1),
            nn.Conv2d(nchannels, nanchors * (5 + nclasses), 1, 1, 0),
        ]
        self.feature = nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x


class YOLOv3MobileSSD(nn.Module):

    def __init__(self, num_classes, in_channels, num_anchors):
        """ Network initialisation """
        super(YOLOv3MobileSSD, self).__init__()
        input_channels = 32
        inplace1 = in_channels[0]
        inplace2 = in_channels[1] + 256
        layer_list = [
            # layer 1, largest
            OrderedDict([
                ('neck_1', HeadBody(inplace1, 512)),
            ]),
            # layer 2
            OrderedDict([
                ('head_1', Head(512, num_anchors[0], num_classes)),
            ]),
            # layer 3
            OrderedDict([
                ('trans_1', Transition(512)),
            ]),

            # layer 4
            # output second scale
            OrderedDict([
                ('neck_2', HeadBody(inplace2, 256)),
            ]),
            # layer 5
            OrderedDict([
                ('head_2', Head(256, num_anchors[1], num_classes)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        head_body1 = self.layers[0](middle_feats[0])
        stage6 = self.layers[1](head_body1)
        trans1 = self.layers[2](head_body1)
        concat2 = torch.cat([trans1, middle_feats[1]], 1)
        head_body2 = self.layers[3](concat2)
        stage5 = self.layers[4](head_body2)

        return stage6, stage5
