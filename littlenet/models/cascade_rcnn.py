import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ._littlenet import Littlenet
from .builder import *
from .. import loss
from ..utils.generate_anchors import *
from ..network.roi import *
from ..utils.bbox import *


class CascadeRCNN(Littlenet):
    def __init__(self, backbone, rpn, roi, head, anchor_cfg, loss_weight=None, neck=None, weights=None, clear=False):
        super(CascadeRCNN, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.rpn_cfg = rpn.pop('cfg')
        self.rpn = build_head(rpn)
        self.num_classes = head.pop('num_classes')
        self.num_stages = len(head)
        self.pooler = Pooler(**roi)
        self.head = nn.ModuleList()
        self.head_cfg = []
        for k, v in head.items():
            v['num_classes'] = self.num_classes
            self.head_cfg.append(v.pop('cfg'))
            self.head.append(build_head(v))
        for k, v in anchor_cfg.items():
            setattr(self, k, v)
        self.loss_weight = [1.] * 2 * (self.num_stages + 1)
        if loss_weight is not None:
            assert len(loss_weight) == 2 * (self.num_stages + 1)
            self.loss_weight = loss_weight

        if weights:
            self.load_weights(weights, clear)
        else:
            self.init_weights()

    def _forward(self, x, target=None):
        feat = self.backbone(x)
        if neck is not None:
            feat = self.neck(feat)
        rpn_obj, rpn_reg = self.rpn(feat)
        all_anchors = []
        for i, p in enumerate(self.pyramid[::-1]):
            anchors = generate_anchors(2 ** (p + 3), self.scales, self.ratios)
            shifts = shift(anchors, feat[i].shape[2:], 2 ** p)
            all_anchors.append(shifts)
        rpnloss = loss.RPNLoss([torch.from_numpy(anchor) for anchor in all_anchors], self.input_size, **self.rpn_cfg)
        ro = rpnloss([rpn_obj, rpn_reg], target)
        rois = ro[0]
        if self.training:
            assert target is not None
            losses = [ro[1] * self.loss_weight[0], ro[2] * self.loss_weight[1]]
            self.log = {'rpn': sum(losses).item()}
            for i in range(self.num_stages):
                lossfn = loss.RoIBoxLoss(self.input_size, **self.head_cfg[i])
                # resample roi
                rois = lossfn.prepare_targets(rois, target)
                x = self.pooler(feat, rois)
                cls, reg = self.head[i](x)
                det, clsloss, regloss = lossfn(cls, reg, rois, target)
                losses += [clsloss * self.loss_weight[2 * i + 2], regloss * self.loss_weight[2 * i + 3]]
                self.log['stage%d' % i] = sum(losses[-2:]).item()
                rois = det
            self.log['tot'] = sum(losses).item()
            return sum(losses)
        result = []
        scores = []
        for i in range(self.num_stages):
            lossfn = loss.RoIBoxLoss(self.input_size, **self.head_cfg[i])
            x = self.pooler(feat, rois)
            cls, reg = self.head[i](x)
            det, res = lossfn(cls, reg, rois)
            result.append(res)
            scores.append(cls)
            rois = det
        cls = sum(scores) / self.num_stages
        # get ensemble result
        result.append(self.getbox(cls, reg, rois, lossfn))
        return result[-1]

    def getbox(self, cls, reg, rois, lossfn):
        device = cls.device
        boxes_per_image = [len(box) for box in rois]
        concat_boxes = torch.cat(rois, dim=0)
        boxes = anchor_decoding(concat_boxes, reg.view(sum(boxes_per_image), -1), lossfn.coder_w).to(device)
        boxes = boxes.split(boxes_per_image, dim=0)
        scores = F.softmax(cls, -1)
        scores = scores.split(boxes_per_image, dim=0)
        result = []
        for box, score in zip(boxes, scores):
            idx = torch.arange(score.size(1), dtype=torch.float, device=device).repeat(score.size(0))
            box = box.repeat(1, score.size(1)).view(-1, 4)
            pred = torch.cat((box, score.view(-1, 1), idx[:, None]), dim=-1)
            pred = clip2img(pred, self.input_size)
            result.append(pred)
        return result

    def load_weights(self, weights_file, clear=False):
        if os.path.splitext(weights_file)[1] != '.pth':
            self.init_weights()
            state = torch.load(weights_file)
            self.seen = 0
            self.backbone.load_state_dict(state)
        else:
            super().load_weights(weights_file, clear)

    # def load_weights(self, weights_file, clear=False):
    #     from ..network.head.roi_box_head import RoIBoxHead
    #     fp = open(weights_file, 'rb')
    #     weights = np.fromfile(fp, dtype=np.float32)
    #     fp.close()
    #
    #     ptr = 0
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             num_w = m.weight.numel()
    #             conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.weight)
    #             m.weight.data.copy_(conv_w)
    #             ptr += num_w
    #             if m.bias is not None:
    #                 num_b = m.bias.numel()
    #                 conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.bias)
    #                 m.bias.data.copy_(conv_b)
    #                 ptr += num_b
    #         if isinstance(m, nn.BatchNorm2d):
    #             num_b = m.weight.numel()
    #             bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.weight)
    #             m.weight.data.copy_(bn_w)
    #             ptr += num_b
    #             bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.bias)
    #             m.bias.data.copy_(bn_b)
    #             ptr += num_b
    #             bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.running_mean)
    #             m.running_mean.data.copy_(bn_rm)
    #             ptr += num_b
    #             bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.running_var)
    #             m.running_var.data.copy_(bn_rv)
    #             ptr += num_b
    #         if isinstance(m, RoIBoxHead):
    #             num_w = m.cls.weight.numel()
    #             lin_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.cls.weight)
    #             m.cls.weight.data.copy_(lin_w)
    #             ptr += num_w
    #             num_b = m.cls.bias.numel()
    #             lin_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.cls.bias)
    #             m.cls.bias.data.copy_(lin_b)
    #             ptr += num_b
    #             num_w = m.reg.weight.numel()
    #             lin_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.reg.weight)
    #             m.reg.weight.data.copy_(lin_w)
    #             ptr += num_w
    #             num_b = m.reg.bias.numel()
    #             lin_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.reg.bias)
    #             m.reg.bias.data.copy_(lin_b)
    #             ptr += num_b
    #             num_w = m.fc6.weight.numel()
    #             lin_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.fc6.weight)
    #             m.fc6.weight.data.copy_(lin_w)
    #             ptr += num_w
    #             num_b = m.fc6.bias.numel()
    #             lin_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.fc6.bias)
    #             m.fc6.bias.data.copy_(lin_b)
    #             ptr += num_b
    #             num_w = m.fc7.weight.numel()
    #             lin_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.fc7.weight)
    #             m.fc7.weight.data.copy_(lin_w)
    #             ptr += num_w
    #             num_b = m.fc7.bias.numel()
    #             lin_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.fc7.bias)
    #             m.fc7.bias.data.copy_(lin_b)
    #             ptr += num_b
    #         if ptr >= len(weights): break
    #     if ptr == len(weights): print('Load success')
