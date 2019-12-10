import torch
import torch.nn as nn
import numpy as np

from .tow_stage import TwoStage


class FasterRCNN(TwoStage):
    def __init__(self, backbone, rpn, roi, head, anchor_cfg, neck=None, weights=None, clear=False):
        rpn_cfg = rpn.pop('cfg')
        head_cfg = head.pop('cfg')
        super(FasterRCNN, self).__init__(backbone, rpn, roi, head, rpn_cfg, head_cfg, neck, weights=weights, clear=clear)

        for k, v in anchor_cfg.items():
            setattr(self, k, v)
