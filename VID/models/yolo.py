import torch
import torch.nn as nn

from littlenet import loss
from littlenet.models._littlenet import Littlenet
from littlenet.models.builder import build_backbone, build_head
from .RNN import CLSTM_cell, CGRU_cell


def build_rnn(cfg):
    assert cfg['type'] in ['ConvLSTM', 'ConvGRU']
    type = cfg.pop('type')
    if type == 'ConvLSTM':
        return CLSTM_cell(**cfg)
    return CGRU_cell(**cfg)


class YOLO(Littlenet):
    def __init__(self, backbone, head, anchors, anchors_mask, rnn=None, ops=None, weights=None, clear=False):
        super(YOLO, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        if rnn is not None:
            self.rnn = build_rnn(rnn)
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.meta = {}
        self.log = {}
        self.num_classes = head['num_classes']

        super().init_weights(weights, clear)

    def _forward(self, inputs, targets=None):
        training = targets is not None
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(0)
            targets = [targets]
        else:
            inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            if targets is not None:
                targets = list(zip(*targets))
            else:
                targets = [None] * inputs.size(0)

        self.log = {}
        outs = []
        for frame, target in zip(inputs, targets):
            outs.append(self.forward_single(frame, target))
        for k in self.log:
            self.log[k] /= inputs.size(0)
        if training:
            return sum(outs) / inputs.size(0)
        else:
            return outs

    def forward_single(self, x, target=None):
        features = self.backbone(x)
        if hasattr(self, 'rnn'):
            features = list(features)
            features[0] = self.rnn(features[0])
            self.meta['rnn'] = self.rnn.hidden_state
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
            log = {'total_loss': loss_tot.item(), 'recall50': r50 / objall, 'iou_avg': iou_sum / objall,
                   'cls_loss': cls_loss.item(), 'reg_loss': reg_loss.item()}
            if len(self.log) == 0:
                self.log.update(log)
            else:
                for k in self.log:
                    self.log[k] += log[k]
            return loss_tot
        else:
            det = []
            for idx in range(len(output)):
                reduction = float(x.shape[2] / output[idx].shape[2])
                lossfn = loss.YOLOLoss(self.num_classes, self.anchors, self.anchors_mask[idx], reduction)
                det.append(lossfn(output[idx]))
            det = torch.cat(det, 1)
            return det

    def clear(self):
        if hasattr(self, 'rnn'):
            self.rnn.clear()
