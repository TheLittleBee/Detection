import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..layer.darknet_layer import Conv2dBatchLeaky


class HeadBody(nn.Module):

    def __init__(self, nchannels, first_head=False):
        super(HeadBody, self).__init__()
        if first_head:
            half_nchannels = int(nchannels / 2)
        else:
            half_nchannels = int(nchannels / 3)
        in_nchannels = 2 * half_nchannels
        layers = [
            Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
            Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
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
            Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
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
            Conv2dBatchLeaky(nchannels, mid_nchannels, 3, 1),
            nn.Conv2d(mid_nchannels, nanchors * (5 + nclasses), 1, 1, 0),
        ]
        self.feature = nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x


class YOLOv3(nn.Module):

    def __init__(self, num_classes, in_channels, num_anchors):
        """ Network initialisation """
        super(YOLOv3, self).__init__()
        input_channels = 32
        inplace1 = in_channels[0]
        inplace2 = in_channels[1] + int(inplace1 / 4)
        inplace3 = in_channels[2] + int(inplace2 / 6)
        layer_list = [
            # layer 1, largest
            OrderedDict([
                ('neck_1', HeadBody(inplace1, first_head=True)),
            ]),
            # layer 2
            OrderedDict([
                ('head_1', Head(int(inplace1 / 2), num_anchors[0], num_classes)),
            ]),
            # layer 3
            OrderedDict([
                ('trans_1', Transition(int(inplace1 / 2))),
            ]),

            # layer 4
            # output second scale
            OrderedDict([
                ('neck_2', HeadBody(inplace2)),
            ]),
            # layer 5
            OrderedDict([
                ('head_2', Head(int(inplace2 / 3), num_anchors[1], num_classes)),
            ]),
            # layer 6
            OrderedDict([
                ('trans_2', Transition(int(inplace2 / 3))),
            ]),

            # layer 7
            # output first scale, smallest
            OrderedDict([
                ('neck_3', HeadBody(inplace3)),
            ]),
            # layer 8 smallest
            OrderedDict([
                ('head_3', Head(int(inplace3 / 3), num_anchors[2], num_classes)),
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
        trans2 = self.layers[5](head_body2)
        concat3 = torch.cat([trans2, middle_feats[2]], 1)
        head_body3 = self.layers[6](concat3)
        stage4 = self.layers[7](head_body3)

        return stage6, stage5, stage4
