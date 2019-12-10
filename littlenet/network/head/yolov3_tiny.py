import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer import darknet_layer as d_layer

__all__ = ['TinyYOLOv3']


class TinyYOLOv3(nn.Module):

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
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        stage6 = self.layers[0](middle_feats[0])
        stage5_upsample = F.interpolate(self.layers[1](middle_feats[0]), scale_factor=2)
        stage5 = self.layers[2](torch.cat((middle_feats[1], stage5_upsample), 1))

        return stage6, stage5
