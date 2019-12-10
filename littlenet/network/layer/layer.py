import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBatch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBatch, self).__init__()

        # Parameters

        if isinstance(kernel_size, (list, tuple)):
            padding = [int(ii / 2) for ii in kernel_size]
        else:
            padding = int(kernel_size / 2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv2dBatchReLU(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dBatchReLU, self).__init__()

        # Parameters

        if isinstance(kernel_size, (list, tuple)):
            padding = [int(ii / 2) for ii in kernel_size]
        else:
            padding = int(kernel_size / 2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
