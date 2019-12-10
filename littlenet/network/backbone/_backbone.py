import torch
import torch.nn as nn
from abc import ABC


class Backbone(ABC):
    def freeze(self, stages):
        if stages > 0:
            for i in range(stages):
                for p in self.layers[i].parameters():
                    p.requires_grad = False
