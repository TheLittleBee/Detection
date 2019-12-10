import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lp':
                pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lse':
                # LSE pool only
                pool = self.logsumexp_2d(x)
            if pool.dim() > 2:
                pool = pool.view(pool.size(0), -1)
            channel_att_raw = self.mlp(pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

    @staticmethod
    def logsumexp_2d(tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7, bn=True, relu=True):
        super(SpatialGate, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(1) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.conv(x_compress)
        if self.bn:
            x_out = self.bn(x_out)
        if self.relu:
            x_out = self.relu(x_out)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
