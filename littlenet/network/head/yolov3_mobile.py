import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..layer.mobilenet_layer import Conv2dDepthWise
from ..layer.layer import Conv2dBatchReLU


class Transition(nn.Module):

    def __init__(self, nchannels):
        super(Transition, self).__init__()
        half_nchannels = int(nchannels / 2)
        layers = [
            nn.Conv2d(nchannels, half_nchannels, 1, 1, bias=False),
            nn.BatchNorm2d(half_nchannels),
            nn.ReLU(inplace=True),
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
            Conv2dDepthWise(nchannels, mid_nchannels, 3, 1),
            nn.Conv2d(mid_nchannels, nanchors * (5 + nclasses), 1, 1, 0),
        ]
        self.feature = nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x


class YOLOv3_Mobile(nn.Module):

    def __init__(self, num_classes, in_channels, num_anchors):
        """ Network initialisation """
        super(YOLOv3_Mobile, self).__init__()

        layer_list = [
            OrderedDict([
                ('neck_1', Conv2dDepthWise(in_channels[0], 256, 3, 1)),
                # ('neck_1', nn.Sequential(
                #     Conv2dBatchReLU(in_channels[0],256,1,1),
                #     Conv2dDepthWise(256, 512, 3, 1),
                #     Conv2dBatchReLU(512,256,1,1),
                #     Conv2dDepthWise(256,512,3,1),
                #     Conv2dBatchReLU(512,256,1,1),
                # )),
            ]),
            OrderedDict([
                ('head_1', Head(256, num_anchors[0], num_classes)),
            ]),
            OrderedDict([
                ('trans_1', Transition(256)),
            ]),

            OrderedDict([
                ('neck_2', Conv2dDepthWise(in_channels[1], 128, 3, 1)),
                # ('neck_2', nn.Sequential(
                #     Conv2dBatchReLU(in_channels[1], 128, 1, 1),
                #     Conv2dDepthWise(128, 256, 3, 1),
                #     Conv2dBatchReLU(256, 128, 1, 1),
                #     Conv2dDepthWise(128, 256, 3, 1),
                #     Conv2dBatchReLU(256, 128, 1, 1),
                # )),
            ]),
            OrderedDict([
                ('head_2', Head(256, num_anchors[1], num_classes)),
            ]),
            # OrderedDict([
            #     ('trans_2', Transition(256)),
            # ]),
            #
            # OrderedDict([
            #     ('neck_3', Conv2dDepthWise(in_channels[2], 128, 3, 1)),
            # ]),
            # OrderedDict([
            #     ('head_3', Head(256, num_anchors[1], num_classes)),
            # ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        neck1 = self.layers[0](middle_feats[0])
        out1 = self.layers[1](neck1)
        trans1 = self.layers[2](neck1)
        neck2 = self.layers[3](middle_feats[1])
        concat1 = torch.cat([trans1, neck2], 1)
        out2 = self.layers[4](concat1)
        # trans2 = self.layers[5](concat1)
        # neck3 = self.layers[6](middle_feats[2])
        # concat2 = torch.cat([trans2, neck3], 1)
        # out3 = self.layers[7](concat2)
        # return out1, out2, out3
        return out1, out2
