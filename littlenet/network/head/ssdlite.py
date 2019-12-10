import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer.mobilenet_layer import InvertedResidual, Conv2dDepthWise


class SSDLite(nn.Module):
    def __init__(self, num_classes, in_channels, num_anchors):
        super(SSDLite, self).__init__()
        self.features = []
        layers = []
        for i in range(2, len(in_channels)):
            layers.append(
                InvertedResidual(in_channels[i - 1], in_channels[i], 2, in_channels[i] // 2 / in_channels[i - 1]))

        self.extra = nn.ModuleList(layers)
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                Conv2dDepthWise(in_channels[i], num_anchors[i] * 4, 3, 1, BN=False))
            cls_convs.append(
                Conv2dDepthWise(in_channels[i], num_anchors[i] * num_classes, 3, 1, BN=False))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

    def forward(self, features):
        self.features.append(features[1])
        self.features.append(features[0])
        x = features[0]
        for l in self.extra:
            x = l(x)
            self.features.append(x)
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(self.features, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return tuple(cls_scores), tuple(bbox_preds)
