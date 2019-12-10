import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.bbox import *
from .utils import smooth_l1_loss

if torch.cuda.is_available():
    from ..utils.sigmoid_focal_loss import sigmoid_focal_loss


class FocalLoss(nn.modules.loss._Loss):
    def __init__(self, anchors, input_size, gamma=2., alpha=.25):
        """"""
        super(FocalLoss, self).__init__()
        self.anchors = anchors  # shape(n,4)
        self.input_size = input_size
        self.gamma = gamma
        self.alpha = alpha
        self.mean = [0, 0, 0, 0]
        self.std = [0.1, 0.1, 0.2, 0.2]
        self.info = {'recall50': 0, 'recall75': 0}

    def forward(self, classification, regression, target=None):
        """

        Arguments:
            classification (list[Tensor(N,A*C,H,W)])
            regression (list[Tensor(N,A*4,H,W)])
            target (list[array(n,5)])
        """
        classification, regression = concat_box_prediction(classification, regression)  # shape(N,-1,C), shape(N,-1,4)
        N, nA, C = classification.size()
        device = classification.device
        self.anchors = self.anchors.to(device)
        pred = anchor_decoding(self.anchors.repeat(N, 1), regression.view(-1, 4))
        pred = pred.view(N, -1, 4)
        if target is None:
            scores = F.softmax(classification, -1)
            result = []
            for box, score in zip(pred, scores):
                idx = torch.arange(score.size(1), dtype=torch.float, device=device).repeat(score.size(0))
                box = box.view(-1, 1, 4).expand(-1, score.size(1), -1).contiguous()
                pred = torch.cat((box.view(-1, 4), score.view(-1, 1), idx[:, None]), dim=-1)
                pred = clip2img(pred, self.input_size)
                result.append(pred)
            return result
        labels, coords = self.prepare_targets(self.anchors, pred, target)
        pos = labels > 0
        sample = labels >= 0
        regloss = smooth_l1_loss(regression[pos], coords[pos], size_average=False) / torch.clamp(pos.sum(), min=1)
        if classification.is_cuda:
            clsloss = sigmoid_focal_loss(classification[sample], labels[sample]).sum() / torch.clamp(sample.sum(), min=1)
        cls_losses = []
        for b in range(N):
            if len(target[b]) == 0:
                continue
            cls = classification[b]
            if not classification.is_cuda:
                label = torch.zeros_like(cls, dtype=torch.float, device=device)
                label[labels[b] == -1, :] = -1
                label[labels[b] > 0, labels[b][labels[b] > 0].long()] = 1
                alpha = torch.full(label.size(), self.alpha, dtype=torch.float, device=device)
                alpha = torch.where(torch.eq(label, 1), alpha, 1. - alpha)
                weight = torch.where(torch.eq(label, 1), 1 - cls, cls)
                weight = alpha * torch.pow(weight, self.gamma)
                cls_loss = F.binary_cross_entropy_with_logits(cls, label, reduction='none')
                cls_loss = weight * cls_loss
                cls_loss = torch.where(torch.ne(label, -1), cls_loss,
                                       torch.zeros(cls_loss.shape, dtype=torch.float, device=device))
                cls_losses.append(cls_loss.sum() / torch.clamp(sample[b].sum(), 1.))

        for k in self.info:
            self.info[k] /= N
        if not classification.is_cuda:
            clsloss = torch.stack(cls_losses).mean()
        return clsloss, regloss

    def prepare_targets(self, anchors, preds, targets):
        """

        Arguments:
            anchors (Tensor(n,4))
            preds (Tensor(N,-1,4))
            targets (list[array(n,5)])
        Returns:
            labels (Tensor(N,n_a))
            coords (Tensor(N,n_a,4))
            pos_mask (Tensor(N,n))
            neg_mask (Tensor(N,n))
        """
        device = anchors.device
        labels = torch.zeros(len(targets), len(anchors), dtype=torch.long, device=device)
        coords = torch.zeros(len(targets), len(anchors), 4, dtype=torch.float, device=device)
        for i, target in enumerate(targets):
            gt = torch.zeros(len(target), 4, dtype=torch.float, device=device)
            for j, anno in enumerate(target):
                gt[j, 0] = (anno[1] - 0.5 * anno[3]) * self.input_size[0]
                gt[j, 1] = (anno[2] - 0.5 * anno[4]) * self.input_size[1]
                gt[j, 2] = (anno[1] + 0.5 * anno[3]) * self.input_size[0]
                gt[j, 3] = (anno[2] + 0.5 * anno[4]) * self.input_size[1]
            iou = bbox_ious(gt, anchors)  # gt*an
            iou_max, iou_idx = iou.max(0)  # an
            below_thr = iou_max < .4
            between_thr = (iou_max >= 0.4) & (iou_max < .5)
            above_thr = iou_max >= .5
            labels[i][between_thr] = -1
            label_gt = torch.from_numpy(target[:, 0]).long().to(device)
            labels[i][above_thr] = label_gt[iou_idx][above_thr]
            match_gt = gt[iou_idx]
            coords[i] = anchor_encoding(anchors, match_gt)
            iou_pred = bbox_ious(preds[i], gt)
            pred_max, pred_idx = iou_pred.max(0)
            self.info['recall50'] += (pred_max >= 0.5).sum().item() / len(target)
            self.info['recall75'] += (pred_max >= 0.75).sum().item() / len(target)

        return labels, coords
