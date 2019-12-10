import torch
import os

from ._littlenet import Littlenet
from .builder import *
from .. import loss
from ..utils.generate_anchors import *
from ..network.roi import *
from ..utils.nms.nms_wrapper import *


class TwoStage(Littlenet):
    def __init__(self, backbone, rpn, roi, head, rpn_cfg, head_cfg, neck=None, mask=None, maskroi=None, weights=None, clear=False):
        super(TwoStage, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.rpn = build_head(rpn)
        self.head = build_head(head)
        self.num_classes = head['num_classes']
        self.pooler = Pooler(**roi)
        if mask is not None:
            self.maskhead = build_head(mask)
            self.maskpooler = Pooler(**maskroi)
            self.masksize = maskroi['resolution']
        self.rpn_cfg = rpn_cfg
        self.head_cfg = head_cfg

        super().init_weights(weights, clear)

    def _forward(self, x, target=None):
        feat = self.backbone(x)
        if hasattr(self, 'neck'):
            feat = self.neck(feat)
        rpn_obj, rpn_reg = self.rpn(feat)
        all_anchors = []
        for i, p in enumerate(self.pyramid[::-1]):
            anchors = generate_anchors(2 ** (p + 3), self.scales, self.ratios)
            shifts = shift(anchors, feat[i].shape[2:], 2 ** p)
            all_anchors.append(shifts)
        rpnloss = loss.RPNLoss([torch.from_numpy(anchor) for anchor in all_anchors], self.input_size, **self.rpn_cfg)
        # get objness topk proposal
        ro = rpnloss([rpn_obj, rpn_reg], target)
        boxloss = loss.RoIBoxLoss(self.input_size, **self.head_cfg)
        if hasattr(self, 'maskhead'):
            masklossfn = loss.MaskLoss(self.masksize)
        roi = ro[0]
        if self.training:
            assert target is not None
            # resample proposals
            roi = boxloss.prepare_targets(roi, target)
            x = self.pooler(feat, roi)
            cls, reg = self.head(x)
            _, clsloss, regloss = boxloss(cls, reg, roi, target)
            self.log = {'rpn_cls': ro[1].item(), 'rpn_reg': ro[2].item(), 'cls_loss': clsloss.item(),
                        'reg_loss': regloss.item()}
            self.log.update(boxloss.info)
            if hasattr(self, 'maskhead'):
                # get pos roi from rpn
                roi = [r[boxloss.pos_mask[i]] for i, r in enumerate(ro[0])]
                x = self.maskpooler(feat, roi)
                maskpred = self.maskhead(x)
                # mask loss
                maskloss = masklossfn(maskpred, roi, boxloss.labels, boxloss.pos_idx, target)
                self.log['mask'] = maskloss.item()
                return clsloss + regloss + ro[1] + ro[2] + maskloss
            return clsloss + regloss + ro[1] + ro[2]
        x = self.pooler(feat, roi)
        cls, reg = self.head(x)
        _, boxes = boxloss(cls, reg, roi)
        if hasattr(self, 'maskhead'):
            # get detection box using nms
            boxes = self.postprogress()
            x = self.maskpooler(feat, boxes)
            # get mask
            mask = self.maskhead(x)
            return boxes, mask
        return boxes
