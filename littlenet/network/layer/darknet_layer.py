import torch
import torch.nn as nn


class Conv2dBatchLeaky(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters

        if isinstance(kernel_size, (list, tuple)):
            padding = [int(ii / 2) for ii in kernel_size]
        else:
            padding = int(kernel_size / 2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),  # , eps=1e-6, momentum=0.01),
            nn.LeakyReLU(leaky_slope, inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class StageBlock(nn.Module):
    custom_layers = ()

    def __init__(self, nchannels):
        super(StageBlock, self).__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),
            Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1)
        )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):
    custom_layers = (StageBlock, StageBlock.custom_layers)

    def __init__(self, nchannels, nblocks, stride=2):
        super(Stage, self).__init__()
        blocks = []
        blocks.append(Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, stride))
        for ii in range(nblocks):
            blocks.append(StageBlock(2 * nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)
