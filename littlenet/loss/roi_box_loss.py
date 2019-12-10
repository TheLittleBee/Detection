import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from ..utils.bbox import *


class RoIBoxLoss(nn.modules.loss._Loss):
    def __init__(self, input_size, num, pos_fraction, pos_iou_thr, neg_iou_thr, coder_w=(10., 10., 5., 5.)):
        """"""
        super(RoIBoxLoss, self).__init__()
        self.input_size = input_size
        self.num = num
        self.pos_fraction = pos_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.coder_w = coder_w
        self.info = {'recall50': 0, 'recall75': 0}

    def forward(self, classification, regression, proposals, target=None):
        """

        Arguments:
            classification (Tensor(np,C))
            regression (Tensor(np,C*4))
            proposals (list[Tensor])
            target (list[array(n,5)])
        """
        device = classification.device
        boxes_per_image = [len(box) for box in proposals]
        concat_boxes = torch.cat(proposals, dim=0)
        boxes = anchor_decoding(concat_boxes, regression.view(sum(boxes_per_image), -1), self.coder_w).to(device)
        boxes = boxes.split(boxes_per_image, dim=0)
        if target is None:
            scores = F.softmax(classification, -1)
            scores = scores.split(boxes_per_image, dim=0)
            result = []
            for box, score in zip(boxes, scores):
                # class agnostic reg 4 each class
                idx = torch.arange(score.size(1), dtype=torch.float, device=device).repeat(score.size(0))
                box = box.repeat(1,score.size(1)).view(-1, 4)
                pred = torch.cat((box, score.view(-1, 1), idx[:, None]), dim=-1)
                pred = clip2img(pred, self.input_size)
                result.append(pred)
            return boxes, result
        if not hasattr(self, 'labels'):
            raise RuntimeError("prepare_targets needs to be called before")
        for k in self.info:
            self.info[k] /= len(proposals)
        labels = self.labels
        coords = self.coords
        cls_loss = F.cross_entropy(classification, labels)
        sampled_pos_inds = torch.nonzero(labels > 0).squeeze(1)
        if regression.size(1) > 4:
            map_inds = 4 * labels[sampled_pos_inds][:, None] + torch.tensor([0, 1, 2, 3], device=device)
            coord_loss = smooth_l1_loss(regression[sampled_pos_inds[:, None], map_inds], coords[sampled_pos_inds],
                                        beta=1.,
                                        size_average=False)
        else:
            coord_loss = smooth_l1_loss(regression[sampled_pos_inds], coords[sampled_pos_inds], beta=1.,
                                        size_average=False)
        return boxes, cls_loss, coord_loss / labels.numel()

    def prepare_targets(self, proposals, targets):
        """

        Arguments:
            proposals (list[Tensor(n,5)]_N)
            preds (list[Tensor(n,nC*4)]_N)
            targets (list[array(n,5)]_N)
        Returns:
            labels (Tensor(np,))
            coords (Tensor(np,4))
            pos_mask (Tensor(np,))
            neg_mask (Tensor(np,))
        """
        labels = []
        coords = []
        self.pos_mask = []
        self.neg_mask = []
        self.gt_idxs = []
        sub_pro = []
        for pro, target in zip(proposals, targets):
            device = pro.device
            gt = torch.zeros(len(target), 4, dtype=torch.float, device=device)
            for j, anno in enumerate(target):
                gt[j, 0] = (anno[1] - 0.5 * anno[3]) * self.input_size[0]
                gt[j, 1] = (anno[2] - 0.5 * anno[4]) * self.input_size[1]
                gt[j, 2] = (anno[1] + 0.5 * anno[3]) * self.input_size[0]
                gt[j, 3] = (anno[2] + 0.5 * anno[4]) * self.input_size[1]
            iou = bbox_ious(gt, pro[:, :4])  # gt*pro
            iou_max, iou_idx = iou.max(0)  # pro
            gt_max, gt_idx = iou.max(dim=1)
            # ensure each gt match its max IoU anchor
            for i in range(len(gt)):
                # if gt_max[i] >= self.neg_iou_thr:
                iou_idx[gt_idx[i]] = i
                iou_max[gt_idx[i]] = self.pos_iou_thr
            below_thr = iou_max < self.neg_iou_thr
            between_thr = (iou_max >= self.neg_iou_thr) & (iou_max < self.pos_iou_thr)
            above_thr = iou_max >= self.pos_iou_thr
            label = torch.zeros(len(pro), dtype=torch.long, device=device)
            label_gt = torch.from_numpy(target[:, 0]).long().to(device)
            label[between_thr] = -1.
            label[above_thr] = label_gt[iou_idx][above_thr]
            match_gt = gt[iou_idx]
            coord = anchor_encoding(pro, match_gt, (10., 10., 5., 5.))
            positive = torch.nonzero(label > 0).squeeze(1)
            negative = torch.nonzero(label == 0).squeeze(1)

            num_pos = int(self.num * self.pos_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.num - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros(
                label.size(0), dtype=torch.uint8, device=device
            )
            neg_idx_per_image_mask = torch.zeros(
                label.size(0), dtype=torch.uint8, device=device
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            self.pos_mask.append(pos_idx_per_image_mask)
            self.neg_mask.append(neg_idx_per_image_mask)
            self.gt_idxs.append(iou_idx)
            sample = pos_idx_per_image_mask | neg_idx_per_image_mask
            labels.append(label[sample])
            coords.append(coord[sample])
            sub_pro.append(pro[sample])
        self.labels = torch.cat(labels, dim=0)
        self.coords = torch.cat(coords, dim=0)
        return sub_pro
