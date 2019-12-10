import torch
import torch.nn as nn
import numpy as np
import time

from .single_stage import SingleStage
from .. import loss
from ..utils.generate_anchors import *


class SSD(SingleStage):
    def __init__(self, backbone, head, strides, anchors_ratios, size_ratios=(0.1, 0.9), weights=None,
                 clear=False):
        self.head_cfg = head.pop('cfg')
        super(SSD, self).__init__(backbone, head, weights=weights, clear=clear)
        self.strides = strides
        self.anchors_ratios = anchors_ratios
        self.size_ratios = size_ratios

    def _forward(self, x, target=None):
        features = self.backbone(x)
        clses, regs = self.head(features)
        # anchor generate
        anchors = self.get_anchors()
        lossfn = loss.SSDLoss(anchors, self.input_size, **self.head_cfg)
        if self.training:
            assert target is not None
            clsloss, regloss = lossfn([clses, regs], target)
            totloss = clsloss + 1. * regloss
            self.log = {'tot_loss': totloss.item(), 'cls_loss': clsloss.item(), 'reg_loss': regloss.item()}
            self.log.update(lossfn.info)
            return totloss
        else:
            return lossfn([clses, regs])

    def get_anchors(self):
        # TODO CHECK SSDLite anchors
        input_size = self.input_size[0]
        min_ratio, max_ratio = self.size_ratios
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(self.anchors_ratios) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if self.size_ratios[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif self.size_ratios[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if self.size_ratios[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif self.size_ratios[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        all_anchors = np.zeros((0, 4), dtype=np.float32)
        for i, r in enumerate(self.anchors_ratios):
            ratios = [1.]
            for rt in r:
                ratios += [1 / rt, rt]
            anchorsa = generate_anchors(min_sizes[i], [1], ratios)
            anchorsb = generate_anchors(np.sqrt(max_sizes[i] * min_sizes[i]), [1.], [1.])
            shifts = shift(np.concatenate((anchorsa, anchorsb)), self.head.features[i].shape[2:], self.strides[i])
            all_anchors = np.append(all_anchors, shifts, 0)
        return torch.from_numpy(all_anchors)
