import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer import darknet_layer as d_layer
from ..plugins.attention import ChannelAttention, SpatialAttention
from ..plugins.non_local import NonLocal2D

__all__ = ['TinyYOLOv3Att']


class TinyYOLOv3Att(nn.Module):

    def __init__(self, num_classes, in_channels, num_anchors):
        """ Network initialisation """
        super().__init__()
        layer_list = [
            # stage 6
            OrderedDict([
                ('1_head', nn.Sequential(d_layer.Conv2dBatchLeaky(256, 512, 3, 1),
                                         nn.Conv2d(512, num_anchors[0] * (5 + num_classes), 1, 1, 0))),
            ]),
            # stage 5
            # stage5 / upsample
            OrderedDict([
                ('trans_1', nn.Sequential(d_layer.Conv2dBatchLeaky(256, 128, 1, 1), )),
            ]),
            # stage5 / head
            OrderedDict([
                ('2_head', nn.Sequential(d_layer.Conv2dBatchLeaky(256 + 128, 256, 3, 1),
                                         nn.Conv2d(256, num_anchors[0] * (5 + num_classes), 1, 1, 0))),
            ]),
            OrderedDict([
                ('satt_1', SpatialAttention(256, 3))
            ]),
            # OrderedDict([
            #     ('satt_1', NonLocal2D(256))
            # ]),
            # OrderedDict([
            #     ('catt_1', ChannelAttention(256, 4))
            # ]),
            OrderedDict([
                ('satt_2', SpatialAttention(256, 3))
            ]),
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])
        self.meta = {}

    def forward(self, middle_feats):
        x = middle_feats[0]
        # x = self.layers[4](middle_feats[0])
        sf, sa = self.layers[3](x)
        self.meta['SA1'] = sa
        # sf = self.layers[3](x)
        x = sf + x
        stage6 = self.layers[0](x)
        stage5_upsample = F.interpolate(self.layers[1](x), scale_factor=2)
        x = middle_feats[1]
        sf, sa = self.layers[4](x)
        self.meta['SA2'] = sa
        x = sf + x
        stage5 = self.layers[2](torch.cat((x, stage5_upsample), 1))

        return stage6, stage5
