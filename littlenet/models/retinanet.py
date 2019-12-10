import torch

from .single_stage import SingleStage
from .. import loss
from ..utils.generate_anchors import *


class RetinaNet(SingleStage):
    def __init__(self, backbone, head, neck, pyramid, anchors_ratios=None, anchors_scales=None, weights=None, clear=False):
        super(RetinaNet, self).__init__(backbone, head, neck, weights=weights, clear=clear)

        self.pyramid = pyramid
        self.ratios = [0.5, 1, 2] if anchors_ratios is None else anchors_ratios
        self.scales = [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)] if anchors_scales is None else anchors_scales

    def _forward(self, x, target=None):
        features = self.backbone(x)
        pyramid = self.neck(features)
        clses, regs = self.head(pyramid)
        if self.exporting: return clses, regs
        all_anchors = np.zeros((0, 4), dtype=np.float32)
        for i, p in enumerate(self.pyramid[::-1]):
            anchors = generate_anchors(2 ** (p + 2), self.scales, self.ratios)
            shifts = shift(anchors, pyramid[i].shape[2:], 2 ** p)
            all_anchors = np.append(all_anchors, shifts, 0)
        lossfunc = loss.FocalLoss(torch.from_numpy(all_anchors.astype(np.float32)), self.input_size)
        if self.training:
            assert target is not None
            clsloss, regloss = lossfunc(clses, regs, target)
            totloss = clsloss + 1. * regloss
            self.log = {'tot_loss': totloss.item(), 'cls_loss': clsloss.item(), 'reg_loss': regloss.item()}
            self.log.update(lossfunc.info)
            return totloss
        else:
            return lossfunc(clses, regs)
