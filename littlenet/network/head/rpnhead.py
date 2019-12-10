import torch
import torch.nn as nn
import torch.nn.functional as F


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(in_channels, num_anchors, 1)
        self.rpn_reg = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, features):
        clses = []
        regs = []
        for feature in features:
            t = F.relu(self.conv(feature))
            clses.append(self.rpn_cls(t))
            regs.append(self.rpn_reg(t))
        return tuple(clses), tuple(regs)
