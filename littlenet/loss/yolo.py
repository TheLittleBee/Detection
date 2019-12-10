import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils.bbox import *


class YOLOLoss(nn.modules.loss._Loss):
    """ Computes yolo loss from darknet network output and target annotation.

    Arguments:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou between a predicted box and ground truth for them to be considered matching
    """

    def __init__(self, num_classes, anchors, anchors_mask, reduction=32, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=1.0, class_scale=1.0, iou_scale=5., thresh=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors_mask)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors) / reduction
        self.anchors_mask = anchors_mask
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.iou_scale = iou_scale
        self.thresh = thresh

        self.info = {'iou_sum': 0, 'recall50': 0, 'recall75': 0,
                     'coord_loss': 0, 'cls_loss': 0}

        # criterion
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.use_iou = False
        self.iou_loss = 0

    def forward(self, output, target=None):
        """ Compute Yolo loss.
        """
        # Parameters
        nB = output.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.size(2)
        nW = output.size(3)
        device = output.device

        self.anchors = self.anchors.to(device)

        # Get x,y,w,h,conf,cls
        output = output.view(nB, nA, -1, nH * nW)
        coordxy = output[:, :, :2].sigmoid()  # tx,ty
        coordwh = output[:, :, 2:4]  # tw,th
        coord = torch.cat((coordxy, coordwh), 2)
        conf = output[:, :, 4].sigmoid()
        if nC > 1:
            cls = output[:, :, 5:].contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)

        # Create prediction boxes
        # time consuming
        lin_x = torch.arange(nW, dtype=torch.float).to(device).repeat(nH, 1).view(nH * nW)
        lin_y = torch.arange(nH, dtype=torch.float).to(device).repeat(nW, 1).t().contiguous().view(nH * nW)
        anchor_w = self.anchors[self.anchors_mask, 0].view(nA, 1).to(device)
        anchor_h = self.anchors[self.anchors_mask, 1].view(nA, 1).to(device)

        pred_boxesx = (coord[:, :, 0] + lin_x).view(-1)
        pred_boxesy = (coord[:, :, 1] + lin_y).view(-1)
        pred_boxesw = (coord[:, :, 2].exp() * anchor_w).view(-1)
        pred_boxesh = (coord[:, :, 3].exp() * anchor_h).view(-1)
        pred_boxes = torch.stack((pred_boxesx, pred_boxesy, pred_boxesw, pred_boxesh), 1)
        # for inference
        if target is None:
            pred_boxes = xywh2xyxy(pred_boxes) * self.reduction
            # det = torch.cat(
            #     (  # x1,y1,x2,y2,obj_conf,cls_conf
            #         pred_boxes.view(nB, -1, 4),
            #         conf.view(nB, -1, 1),
            #         F.softmax(cls, 1).view(nB, -1, nC)
            #     ), -1
            # )
            score = F.softmax(cls, 1).view(nB, -1, nC) * conf.view(nB, -1, 1)
            # score = F.softmax(cls.view(nB, -1, nC) * conf.view(nB, -1, 1), -1)
            idx = torch.arange(nC).repeat(nB, score.size(1)).float().to(device)
            pred_boxes = pred_boxes.repeat(1, nC)
            # x1,y1,x2,y2,score,idx     repeat cls agnostic boxes
            det = torch.cat((pred_boxes.view(nB, -1, 4), score.view(nB, -1, 1), idx.view(nB, -1, 1)), -1)
            return det
        # Get target values
        coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(
            pred_boxes, conf.view(-1, 1), torch.argmax(cls, 1).view(-1, 1), target, nH, nW)
        # coord
        coord_mask = coord_mask.expand_as(tcoord)[:, :, :2]  # 0 = 1 = 2 = 3, only need first two element
        coord_center, tcoord_center = coord[:, :, :2], tcoord[:, :, :2]
        coord_wh, tcoord_wh = coord[:, :, 2:], tcoord[:, :, 2:]
        if nC > 1:
            tcls = tcls[cls_mask].view(-1).long()

            cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
            cls = cls[cls_mask].view(-1, nC)

        # criteria
        self.bce = self.bce.to(device)
        self.mse = self.mse.to(device)
        self.smooth_l1 = self.smooth_l1.to(device)
        self.ce = self.ce.to(device)

        bce = self.bce
        mse = self.mse
        smooth_l1 = self.smooth_l1
        ce = self.ce

        # Compute losses
        if self.use_iou:
            self.iou_loss *= self.iou_scale
            self.loss_coord = self.iou_loss / nB
        else:
            loss_coord_center = 2.0 * 1.0 * self.coord_scale * (coord_mask * bce(coord_center, tcoord_center)).sum()
            loss_coord_wh = 2.0 * 1.5 * self.coord_scale * (coord_mask * smooth_l1(coord_wh, tcoord_wh)).sum()
            self.loss_coord = (loss_coord_center + loss_coord_wh) / nB  # / torch.clamp((coord_mask > 0).sum()/2, min=1)

        loss_conf_pos = 1.0 * self.object_scale * (conf_pos_mask * bce(conf, tconf)).sum()  # / torch.clamp(conf_pos_mask.sum(), min=1)
        loss_conf_neg = 1.0 * self.noobject_scale * (conf_neg_mask * bce(conf, tconf)).sum()  # / conf_neg_mask.sum()
        self.loss_conf = (loss_conf_pos + loss_conf_neg) / nB

        if nC > 1 and cls.numel() > 0:
            self.loss_cls = self.class_scale * 1.0 * ce(cls, tcls) / nB
        else:
            self.loss_cls = torch.tensor(0.0, device=device)

        self.info['coord_loss'] = self.loss_coord
        self.info['cls_loss'] = self.loss_cls

        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
        return self.loss_tot

    def build_targets(self, pred_boxes, conf, cls, ground_truth, nH, nW):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors

        Arguments:
            pred_boxes: (nB * nA * nH * nW,4)
            conf: (nB * nA * nH * nW,1)
            cls: (nB * nA * nH * nW,1)
        Returns:
            coord_mask: (nB,nA,1,nH*nW) 2-anno.w*anno.h
            conf_pos_mask: (nB,nA,nH*nW) where anno
            conf_neg_mask: (nB,nA,nH*nW) where no anno
            cls_mask: (nB,nA,nH*nW) type=uint8 where anno
            tcoord: (nB,nA,4,nH*nW) target
            tconf: (nB,nA,nH*nW) where anno
            tcls: (nB,nA,nH*nW) anno.label
        """
        # Parameters
        nB = len(ground_truth)
        nA = self.num_anchors
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        device = pred_boxes.device

        # Tensors
        conf_pos_mask = torch.zeros(nB, nA, nH * nW, requires_grad=False, device=device)
        conf_neg_mask = torch.ones(nB, nA, nH * nW, requires_grad=False, device=device)
        coord_mask = torch.zeros(nB, nA, 1, nH * nW, requires_grad=False, device=device)
        cls_mask = torch.zeros(nB, nA, nH * nW, requires_grad=False, dtype=torch.uint8, device=device)
        tcoord = torch.zeros(nB, nA, 4, nH * nW, requires_grad=False, device=device)
        tconf = torch.zeros(nB, nA, nH * nW, requires_grad=False, device=device)
        tcls = torch.zeros(nB, nA, nH * nW, requires_grad=False, device=device)

        recall50 = 0
        recall75 = 0
        iou_sum = 0
        for b in range(nB):
            if len(ground_truth[b]) == 0:  # No gt for this image
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors]
            # cur_pred_conf = conf[b * nAnchors:(b + 1) * nAnchors]
            # cur_pred_cls = cls[b * nAnchors:(b + 1) * nAnchors]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 4, device=device, dtype=torch.float)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = anno[1] * nW
                gt[i, 1] = anno[2] * nH
                gt[i, 2] = anno[3] * nW
                gt[i, 3] = anno[4] * nH

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes, xywh=True)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_neg_mask[b][mask.view_as(conf_neg_mask[b])] = 0

            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors, xywh=True)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each gt
            # time consuming
            for i, anno in enumerate(ground_truth[b]):
                gi = min(nW - 1, max(0, int(gt[i, 0])))
                gj = min(nH - 1, max(0, int(gt[i, 1])))
                cur_n = best_anchors[i]
                if cur_n in self.anchors_mask:
                    best_n = self.anchors_mask.index(cur_n)
                else:
                    continue

                iou = iou_gt_pred[i][best_n * nPixels + gj * nW + gi]
                # IoU loss
                # self.iou_loss += -iou.clamp(1e-6).log()
                self.iou_loss += 1 - iou
                # debug information
                recall50 += (iou > 0.5).item()
                recall75 += (iou > 0.75).item()
                iou_sum += iou.item()

                coord_mask[b][best_n][0][gj * nW + gi] = 2 - anno[3] * anno[4]
                cls_mask[b][best_n][gj * nW + gi] = 1
                conf_pos_mask[b][best_n][gj * nW + gi] = 1
                conf_neg_mask[b][best_n][gj * nW + gi] = 0
                tcoord[b][best_n][0][gj * nW + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * nW + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * nW + gi] = math.log(gt[i, 2] / self.anchors[cur_n, 0])
                tcoord[b][best_n][3][gj * nW + gi] = math.log(gt[i, 3] / self.anchors[cur_n, 1])
                tconf[b][best_n][gj * nW + gi] = 1
                tcls[b][best_n][gj * nW + gi] = anno[0]
                # if cur_pred_cls[gj * nW + gi] == anno[0]:  # and cur_pred_conf[gj * nW + gi] > .5:
                #     precision50 += (iou > 0.5).item()
                #     precision75 += (iou > 0.75).item()
        # obj_pro = conf_pos_mask[conf.view(nB, nA, nH * nW) > .5].sum().item()
        # loss informaion
        self.info['iou_sum'] = iou_sum
        self.info['recall50'] = recall50
        self.info['recall75'] = recall75

        return coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, tcoord, tconf, tcls
