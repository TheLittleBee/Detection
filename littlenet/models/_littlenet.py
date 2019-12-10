import os
import torch
import torch.nn as nn
import numpy as np
import abc


class Littlenet(nn.Module):

    def __init__(self):
        super(Littlenet, self).__init__()
        self.input_size = (416, 416)

        self.seen = 0
        self.exporting = False

    @abc.abstractmethod
    def _forward(self, x, target=None):
        pass

    def export(self, x, file):
        self.exporting = True
        torch.onnx.export(self, x, file, verbose=True)
        self.exporting = False

    def forward(self, x, target=None):
        if self.training:
            self.seen += x.size(0)
            return self._forward(x, target)
        else:
            return self._forward(x)

    def init_weights(self, weights=None, clear=False, mode='fan_in', slope=0.1):
        for md in self.children():
            if type(md) == type(self.backbone): continue
            for m in md.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        if weights is None:
            if not max([r in self.backbone.__class__.__name__ for r in ['Res', 'Shuffle', 'Inception']]):
                raise Exception('Not support backbone from scratch!!!')
        elif os.path.splitext(weights)[-1] == '.pth':
            self.load_weights(weights, clear=clear)
        else:
            self.backbone.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)

    def load_weights(self, weights_file, clear=False):
        state = torch.load(weights_file, map_location='cpu')
        self.seen = 0 if clear else state['seen']
        self.load_state_dict(state['weights'])

    def save_weights(self, weights_file):
        state = {
            'seen': self.seen,
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)
