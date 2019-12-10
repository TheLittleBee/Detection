import torch.nn as nn
from collections import OrderedDict

from ..layer import mobilenet_layer as m_layer
from ..layer import layer as l_layer
from ._backbone import Backbone


class MobileNet(nn.Module, Backbone):
    def __init__(self, alpha=1.):
        super(MobileNet, self).__init__()

        layer_list = [
            OrderedDict([
                ('Conv1', nn.Sequential(
                    l_layer.Conv2dBatchReLU(3, int(alpha * 32), 3, 2),
                    m_layer.Conv2dDepthWise(int(alpha * 32), int(alpha * 64), 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv2', nn.Sequential(
                    m_layer.Conv2dDepthWise(int(alpha * 64), int(alpha * 128), 3, 2),
                    m_layer.Conv2dDepthWise(int(alpha * 128), int(alpha * 128), 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv3', nn.Sequential(
                    m_layer.Conv2dDepthWise(int(alpha * 128), int(alpha * 256), 3, 2),
                    m_layer.Conv2dDepthWise(int(alpha * 256), int(alpha * 256), 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv4', nn.Sequential(
                    m_layer.Conv2dDepthWise(int(alpha * 256), int(alpha * 512), 3, 2),
                    m_layer.Conv2dDepthWise(int(alpha * 512), int(alpha * 512), 3, 1),
                    m_layer.Conv2dDepthWise(int(alpha * 512), int(alpha * 512), 3, 1),
                    m_layer.Conv2dDepthWise(int(alpha * 512), int(alpha * 512), 3, 1),
                    m_layer.Conv2dDepthWise(int(alpha * 512), int(alpha * 512), 3, 1),
                    m_layer.Conv2dDepthWise(int(alpha * 512), int(alpha * 512), 3, 1),
                )),
            ]),
            OrderedDict([
                ('Conv5', nn.Sequential(
                    m_layer.Conv2dDepthWise(int(alpha * 512), int(alpha * 1024), 3, 2),
                    m_layer.Conv2dDepthWise(int(alpha * 1024), int(alpha * 1024), 3, 1),
                )),
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        stage1 = self.layers[0](x)
        stage2 = self.layers[1](stage1)
        stage3 = self.layers[2](stage2)
        stage4 = self.layers[3](stage3)
        stage5 = self.layers[4](stage4)

        return stage5, stage4, stage3
