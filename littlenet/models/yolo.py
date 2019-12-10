import torch
import torch.nn as nn
import numpy as np
import time

from .single_stage import SingleStage
from .. import loss
from ..utils.bbox import clip2img
from ..utils.generate_anchors import shift


class YOLO(SingleStage):
    def __init__(self, backbone, head, anchors, anchors_mask, weights=None, clear=False):
        super(YOLO, self).__init__(backbone, head, weights=weights, clear=clear)
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.meta = {}

    def _forward(self, x, target=None):
        features = self.backbone(x)
        output = self.head(features)
        if hasattr(self.head, 'meta'):
            self.meta.update(self.head.meta)
        if self.exporting: return output
        if self.training:
            assert target is not None
            loss_tot = cls_loss = reg_loss = 0
            r50 = r75 = objall = iou_sum = 0
            for gt in target:
                objall += len(gt)
            if not objall: objall = 1
            for idx in range(len(output)):
                reduction = float(x.shape[2] / output[idx].shape[2])
                lossfn = loss.YOLOLoss(self.num_classes, self.anchors, self.anchors_mask[idx], reduction)
                loss_tot += lossfn(output[idx], target)
                info = lossfn.info
                r50 += info['recall50']
                r75 += info['recall75']
                cls_loss += info['cls_loss']
                reg_loss += info['coord_loss']
                iou_sum += info['iou_sum']
            self.log = {'total_loss': loss_tot.item(), 'recall50': r50 / objall, 'iou_avg': iou_sum / objall,
                        'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item()}
            return loss_tot
        else:
            det = []
            for idx in range(len(output)):
                reduction = float(x.shape[2] / output[idx].shape[2])
                lossfn = loss.YOLOLoss(self.num_classes, self.anchors, self.anchors_mask[idx], reduction)
                det.append(lossfn(output[idx]))
            det = torch.cat(det, 1)
            return tuple(clip2img(d, self.input_size, remove_empty=True) for d in det)

    # def _forward(self, x, target=None):
    #     features = self.backbone(x)
    #     output = self.head(features)
    #     if self.exporting: return output
    #     anchors = self.get_anchors(features)
    #     lossfn = loss.YOLOAnchor(anchors, self.anchors_mask, self.input_size)
    #     if self.training:
    #         assert target is not None
    #         cls_loss, reg_loss, conf_loss = lossfn(output, target)
    #         loss_tot = cls_loss + 2 * reg_loss + conf_loss
    #         # loss_tot = cls_loss + 10 * reg_loss + conf_loss
    #         self.log = {'total_loss': loss_tot.item(),
    #                     'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item()}
    #         return loss_tot
    #     else:
    #         return lossfn(output)
    #
    # def get_anchors(self, feats):
    #     all_anchors = np.zeros((0, 4), dtype=np.float32)
    #     for i, mask in enumerate(self.anchors_mask):
    #         anchor = []
    #         for m in mask:
    #             anchor.append([1, 1] + self.anchors[m])
    #         reduction = self.input_size[0] / feats[i].shape[-1]
    #         shifts = shift(np.array(anchor), feats[i].shape[-2:], reduction)
    #         all_anchors = np.append(all_anchors, shifts, 0)
    #     return torch.from_numpy(all_anchors).float()
