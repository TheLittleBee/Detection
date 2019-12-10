import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .anchor_based import AnchorLoss
from .utils import *
from ..utils.bbox import *


class SSDLoss(AnchorLoss):
    """
    """

    def __init__(self, anchors, input_size, pos_iou_thr, neg_iou_thr, neg_pos_ratio):
        super(SSDLoss, self).__init__(anchors, input_size, pos_iou_thr, neg_iou_thr)
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, output, target=None):
        assert isinstance(output, (list, tuple)) and len(output) == 2
        classification, regression = output
        classification, regression = concat_box_prediction(classification, regression)  # shape(N,-1,C), shape(N,-1,4)
        N, nA, C = classification.size()
        device = classification.device
        self.anchors = self.anchors.to(device)
        boxes = anchor_decoding(self.anchors.repeat(N, 1), regression.view(-1, 4))
        boxes = boxes.view(N, -1, 4)
        if target is None:
            scores = F.softmax(classification, -1)
            result = []
            for box, score in zip(boxes, scores):
                idx = torch.arange(score.size(1), dtype=torch.float, device=device).repeat(score.size(0))
                box = box.view(-1, 1, 4).expand(-1, score.size(1), -1).contiguous()
                pred = torch.cat((box.view(-1, 4), score.view(-1, 1), idx[:, None]), dim=-1)
                pred = clip2img(pred, self.input_size)
                result.append(pred)
            return result
        labels, coords = self.match_targets(target)
        pos_mask, neg_mask = self.resample(labels)
        sample_mask = pos_mask | neg_mask
        coord_loss = smooth_l1_loss(regression[pos_mask], coords[pos_mask], beta=1., size_average=False)
        cls_loss_pos = F.cross_entropy(classification[pos_mask], labels[pos_mask], reduction='none')
        # resample by topk neg loss
        num_pos = pos_mask.sum()
        num_neg = self.neg_pos_ratio * num_pos
        cls_loss_neg, _ = F.cross_entropy(classification[neg_mask], labels[neg_mask], reduction='none').topk(num_neg)
        cls_loss = cls_loss_pos.sum() + cls_loss_neg.sum()
        return cls_loss / num_pos, coord_loss / num_pos
