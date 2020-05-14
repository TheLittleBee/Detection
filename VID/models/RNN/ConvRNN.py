import torch
import torch.nn as nn


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """

    def __init__(self, in_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.input_channels = in_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))
        self.hidden_state = None

    def forward(self, inputs):
        device = inputs.device
        if self.hidden_state is None:
            htprev = torch.zeros(inputs.size(0), self.num_features,
                                 inputs.size(2), inputs.size(3)).to(device)
        else:
            htprev = self.hidden_state
        x = inputs

        combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
        gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

        zgate, rgate = torch.split(gates, self.num_features, dim=1)
        # zgate, rgate = gates.chunk(2, 1)
        z = torch.sigmoid(zgate)
        r = torch.sigmoid(rgate)

        combined_2 = torch.cat((x, r * htprev),
                               1)  # h' = tanh(W*(x+r*H_t-1))
        ht = self.conv2(combined_2)
        ht = torch.tanh(ht)
        htnext = (1 - z) * htprev + z * ht
        self.hidden_state = htnext
        return htnext

    def clear(self):
        self.hidden_state = None


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, in_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.input_channels = in_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))
        self.forget_state = None
        self.hidden_state = None

    def forward(self, inputs):
        device = inputs.device
        if self.hidden_state is None:
            hx = torch.zeros(inputs.size(0), self.num_features, inputs.size(2), inputs.size(3)).to(device)
        else:
            hx = self.hidden_state
        if self.forget_state is None:
            cx = torch.zeros(inputs.size(0), self.num_features, inputs.size(2), inputs.size(3)).to(device)
        else:
            cx = self.forget_state
        x = inputs

        combined = torch.cat((x, hx), 1)
        gates = self.conv(combined)  # gates: S, num_features*4, H, W
        # it should return 4 tensors: i,f,g,o
        ingate, forgetgate, cellgate, outgate = torch.split(
            gates, self.num_features, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        self.forget_state = cy
        self.hidden_state = hy
        return hy

    def clear(self):
        self.forget_state = None
        self.hidden_state = None
