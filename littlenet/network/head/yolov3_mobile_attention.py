import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..layer.mobilenet_layer import Conv2dDepthWise
from ..plugins.attention import ChannelAttention


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


class Attention(nn.Module):
    def __init__(self, inp, oup=1):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = self.feature(data)
        return data * x
        # return x


class YOLOv3_Mobile_Att(nn.Module):

    def __init__(self, num_classes, in_channels, num_anchors):
        """ Network initialisation """
        super(YOLOv3_Mobile_Att, self).__init__()

        layer_list = [
            OrderedDict([
                ('neck_1', Conv2dDepthWise(in_channels[0], 256, 3, 1)),
            ]),
            OrderedDict([
                ('att_1', Attention(256, 1)),
            ]),
            # OrderedDict([
            #     ('catt_1', ChannelAttention(256, 4)),
            # ]),
            OrderedDict([
                ('head_1', Head(256, num_anchors[0], num_classes)),
            ]),
            OrderedDict([
                ('trans_1', Transition(256)),
            ]),

            OrderedDict([
                ('neck_2', Conv2dDepthWise(in_channels[1], 128, 3, 1)),
            ]),
            OrderedDict([
                ('att_2', Attention(256, 1)),
                # ('att_2', Attention(128, 1)),
            ]),
            # OrderedDict([
            #     ('catt_2', ChannelAttention(256, 4)),
            # ]),
            OrderedDict([
                ('head_2', Head(256, num_anchors[1], num_classes)),
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        # neck1 = self.layers[0](middle_feats[0])
        # att1 = self.layers[1](neck1)
        # catt1 = self.layers[2](att1)
        # out1 = self.layers[3](catt1)
        # trans1 = self.layers[4](neck1)
        # neck2 = self.layers[5](middle_feats[1])
        # concat1 = torch.cat([trans1, neck2], 1)
        # att2 = self.layers[6](concat1)
        # catt2 = self.layers[7](att2)
        # out2 = self.layers[8](catt2)

        neck1 = self.layers[0](middle_feats[0])
        att1 = self.layers[1](neck1)  # * neck1
        out1 = self.layers[2](att1)
        trans1 = self.layers[3](neck1)
        neck2 = self.layers[4](middle_feats[1])
        concat1 = torch.cat([trans1, neck2], 1)
        # att2 = self.layers[5](neck2)
        # att2 = att2 * concat1
        att2 = self.layers[5](concat1)
        out2 = self.layers[6](att2)
        return out1, out2
