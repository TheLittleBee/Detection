import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.bbox import *
from .utils import *


class AnchorLoss(nn.modules.loss._Loss):
    def __init__(self, anchors, input_size, pos_iou_thr, neg_iou_thr, coder_w=(10., 10., 5., 5.)):
        """

        """
        super(AnchorLoss, self).__init__()
        self.anchors = anchors
        self.input_size = input_size
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.coder_w = coder_w
        self.info = {'recall50': 0, 'recall75': 0}

    def forward(self, output, target=None):
        """

        Arguments:
            classification (list[Tensor]): low resolution first
            regression (list[Tensor]): low resolution first, class agnostic
            target (list[array(n,5)])
        """
        assert isinstance(output, (list, tuple)) and len(output) == 2
        classification, regression = output
        classification, regression = concat_box_prediction(classification, regression)  # shape(N,-1,C), shape(N,-1,4)
        N, nA, C = classification.size()
        device = classification.device
        self.anchors = self.anchors.to(device)
        boxes = anchor_decoding(self.anchors.repeat(N, 1), regression.view(-1, 4), self.coder_w)
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
        coord_loss = smooth_l1_loss(regression[pos_mask], coords[pos_mask], beta=1., size_average=True)
        cls_loss = F.cross_entropy(classification[sample_mask], labels[sample_mask])
        return cls_loss, coord_loss

    def match_targets(self, targets):
        """

        Arguments:
            targets (list[array(nt,5)])
        Returns:
            labels (Tensor(N,n_a)): dtype=torch.long
            coords (Tensor(N,n_a,4))
        """
        anchors = self.anchors
        device = anchors.device
        labels = torch.zeros(len(targets), len(anchors), dtype=torch.long, device=device)
        coords = torch.zeros(len(targets), len(anchors), 4, dtype=torch.float, device=device)
        gts = torch.zeros(len(targets), len(anchors), 4, dtype=torch.float, device=device)
        for i, target in enumerate(targets):
            gt = torch.zeros(len(target), 4, dtype=torch.float, device=device)
            for j, anno in enumerate(target):
                gt[j, 0] = (anno[1] - 0.5 * anno[3]) * self.input_size[0]
                gt[j, 1] = (anno[2] - 0.5 * anno[4]) * self.input_size[1]
                gt[j, 2] = (anno[1] + 0.5 * anno[3]) * self.input_size[0]
                gt[j, 3] = (anno[2] + 0.5 * anno[4]) * self.input_size[1]
            # match strategy
            iou = bbox_ious(gt, anchors)  # gt*an
            iou_max, iou_idx = iou.max(0)  # an
            gt_max, gt_idx = iou.max(dim=1)
            # ensure each gt match its max IoU anchor
            for k in range(len(gt)):
                # if gt_max[i] >= self.neg_iou_thr:
                iou_idx[gt_idx[k]] = k
                iou_max[gt_idx[k]] = self.pos_iou_thr
            below_thr = iou_max < self.neg_iou_thr
            between_thr = (iou_max >= self.neg_iou_thr) & (iou_max < self.pos_iou_thr)
            above_thr = iou_max >= self.pos_iou_thr
            label_gt = torch.from_numpy(target[:, 0]).long().to(device)
            labels[i][between_thr] = -1.
            labels[i][above_thr] = label_gt[iou_idx][above_thr]
            match_gt = gt[iou_idx]
            coords[i] = anchor_encoding(anchors, match_gt, self.coder_w)
            gts[i] = match_gt
        self.gts = gts

        return labels, coords

    def resample(self, labels):
        return labels > 0, labels == 0
