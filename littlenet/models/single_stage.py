import torch.nn as nn
import torch
import os
import numpy as np

from ._littlenet import Littlenet
from .builder import *


class SingleStage(Littlenet):
    def __init__(self, backbone, head, neck=None, weights=None, clear=None):
        super(SingleStage, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.head = build_head(head)
        self.num_classes = head['num_classes']

        super().init_weights(weights, clear)

    def _forward(self, x, target=None):
        pass
