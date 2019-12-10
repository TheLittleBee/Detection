import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_based import AnchorLoss
from ..utils.bbox import *
from .utils import *


class YOLOAnchor(AnchorLoss):
    def __init__(self, anchors, anchors_mask, input_size, pos_iou_thr=0.5, neg_iou_thr=0.5, neg_pos_ratio=3,
                 coder_w=(10., 10., 5., 5.)):
        super(YOLOAnchor, self).__init__(anchors, input_size, pos_iou_thr, neg_iou_thr, coder_w)
        self.anchors_mask = anchors_mask
        self.neg_pos_ratio = neg_pos_ratio
        self.use_iou = False

    def forward(self, output, target=None):
        output = self.concat_output(output)
        reg = output[..., :4]
        conf = output[..., 4].sigmoid()
        cls = output[..., 5:]
        N, nA, C = cls.size()
        device = cls.device
        self.anchors = self.anchors.to(device)
        boxes = anchor_decoding(self.anchors.repeat(N, 1), reg.view(-1, 4), self.coder_w)
        boxes = boxes.view(N, -1, 4)
        if target is None:
            scores = F.softmax(cls, -1) * conf.view(N, -1, 1)
            result = []
            for box, score in zip(boxes, scores):
                idx = torch.arange(score.size(1), dtype=torch.float, device=device).repeat(score.size(0))
                box = box.view(-1, 1, 4).expand(-1, score.size(1), -1).contiguous()
                pred = torch.cat((box.view(-1, 4), score.view(-1, 1), idx[:, None]), dim=-1)
                pred = clip2img(pred, self.input_size)
                result.append(pred)
            return tuple(result)
        for t in target:
            t[:, 0] += 1
        labels, coords = self.match_targets(target)
        pos_mask, neg_mask = self.resample(labels)
        sample_mask = pos_mask | neg_mask
        if self.use_iou:
            coord_loss = iou_loss(boxes[pos_mask], self.gts[pos_mask], giou=True).sum()
        else:
            coord_loss = smooth_l1_loss(reg[pos_mask], coords[pos_mask], size_average=False)
        cls_loss = F.cross_entropy(cls[pos_mask], labels[pos_mask] - 1, reduction='sum')
        # conf loss
        tconf = pos_mask.float()
        conf_loss_pos = F.binary_cross_entropy(conf[pos_mask], tconf[pos_mask], reduction='none')
        num_pos = pos_mask.sum()
        num_neg = self.neg_pos_ratio * num_pos
        conf_loss_neg, _ = F.binary_cross_entropy(conf[neg_mask], tconf[neg_mask], reduction='none').topk(num_neg)
        conf_loss = conf_loss_pos.sum() + conf_loss_neg.sum()
        return cls_loss, coord_loss, conf_loss

    def concat_output(self, output):
        out = []
        for t, m in zip(output, self.anchors_mask):
            N, mC, H, W = t.size()
            C = mC // len(m)
            out.append(t.view(N, len(m), C, H, W).permute(0, 3, 4, 1, 2).contiguous().view(N, -1, C))
        return torch.cat(out, dim=1)
