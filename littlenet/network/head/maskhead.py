import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    def __init__(self, num_conv, resolution, in_channel, out_channel, num_classes):
        super(MaskHead, self).__init__()
        conv = []
        for i in range(num_conv):
            conv.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
            conv.append(nn.ReLU(inplace=True))
            in_channel = out_channel
        self.conv = nn.Sequential(*conv)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.logit = nn.Conv2d(out_channel, num_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.logit(x)
        return x
