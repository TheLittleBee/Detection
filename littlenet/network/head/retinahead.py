import torch.nn as nn
import torch.nn.functional as F
from ..layer.mobilenet_layer import Conv2dDepthWise


class RetinaHead(nn.Module):
    # TODO check class agnostic box regression
    def __init__(self, num_classes, num_anchors, inp=256, feat=256, stack=4, mobile=False):
        """"""
        super(RetinaHead, self).__init__()
        cls = []
        reg = []
        for i in range(stack):
            ch = inp if i == 0 else feat
            if mobile:
                cls.append(Conv2dDepthWise(ch, feat, 3, 1, BN=False))
                cls.append(nn.ReLU(inplace=True))
                reg.append(Conv2dDepthWise(ch, feat, 3, 1, BN=False))
                reg.append(nn.ReLU(inplace=True))
            else:
                cls.append(nn.Conv2d(ch, feat, 3, 1, 1))
                cls.append(nn.ReLU(inplace=True))
                reg.append(nn.Conv2d(ch, feat, 3, 1, 1))
                reg.append(nn.ReLU(inplace=True))
        if mobile:
            cls.append(Conv2dDepthWise(feat, num_anchors * num_classes, 3, 1, BN=False))
            reg.append(Conv2dDepthWise(feat, num_anchors * 4, 3, 1, BN=False))
        else:
            cls.append(nn.Conv2d(feat, num_anchors * num_classes, 3, 1, 1))
            reg.append(nn.Conv2d(feat, num_anchors * 4, 3, 1, 1))
        self.cls = nn.Sequential(*cls)
        self.reg = nn.Sequential(*reg)

    def forward(self, x):
        cls = []
        reg = []
        for feature in x:
            cls.append(self.cls(feature))
            reg.append(self.reg(feature))
        return tuple(cls), tuple(reg)
