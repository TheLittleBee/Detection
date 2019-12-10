import torch.nn as nn
import torch.nn.functional as F

from ..layer.mobilenet_layer import Conv2dDepthWise


class FPN(nn.Module):
    def __init__(self, inps, oup, extra_num=2, extra_conv=True, on_inp=True, norm=None, mobile=False):
        super(FPN, self).__init__()
        self.extra_num = extra_num
        self.on_input = on_inp
        self.laterals = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(inps)):
            self.laterals.append(nn.Conv2d(inps[i], oup, 1, 1, 0))
            if mobile:
                self.fpn_convs.append(Conv2dDepthWise(oup, oup, 3, 1, BN=False))
            else:
                self.fpn_convs.append(nn.Conv2d(oup, oup, 3, 1, 1))

        if extra_num:
            self.extra = nn.ModuleList()
            for i in range(extra_num):
                if extra_conv:
                    if i == 0 and on_inp:
                        if mobile:
                            self.extra.append(Conv2dDepthWise(inps[-1], oup, 3, 2, BN=False))
                        else:
                            self.extra.append(nn.Conv2d(inps[-1], oup, 3, 2, 1))
                    else:
                        if mobile:
                            self.extra.append(Conv2dDepthWise(oup, oup, 3, 2, BN=False))
                        else:
                            self.extra.append(nn.Conv2d(oup, oup, 3, 2, 1))
                else:
                    self.extra.append(nn.MaxPool2d(1, 2, 0))

    def forward(self, features):
        '''top down order'''
        laterals = [conv(features[i]) for i, conv in enumerate(self.laterals[::-1])]
        num = len(laterals)
        for i in range(num - 1):
            laterals[i + 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        outs = [self.fpn_convs[::-1][i](laterals[i]) for i in range(len(laterals))]
        if self.extra_num:
            if self.on_input:
                outs.insert(0, self.extra[0](features[0]))
            else:
                outs.insert(0, self.extra[0](outs[0]))
            for i in range(1, self.extra_num):
                outs.insert(0, self.extra[i](outs[0]))
        return tuple(outs)
