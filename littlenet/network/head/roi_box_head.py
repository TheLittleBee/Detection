import torch
from torch import nn
import torch.nn.functional as F


class RoIBoxHead(nn.Module):
    # TODO ADD class specific box regression
    def __init__(self, in_channels, resolution, fc_size, num_classes):
        super(RoIBoxHead, self).__init__()
        inp = in_channels * resolution ** 2
        self.fc6 = nn.Linear(inp, fc_size)
        self.fc7 = nn.Linear(fc_size, fc_size)
        self.cls = nn.Linear(fc_size, num_classes)
        # self.reg = nn.Linear(fc_size, num_classes * 4)
        self.reg = nn.Linear(fc_size, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        cls = self.cls(x)
        reg = self.reg(x)
        return cls, reg
