import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from ..utils.bbox import *


class MaskLoss(nn.modules.loss._Loss):
    def __init__(self, mask_size):
        super(MaskLoss, self).__init__()
        self.mask_size = mask_size

    def forward(self, mask_pred, proposals,label,pos_idx, target):
        mask_target = self.prepare_target(proposals,pos_idx,target)
        pos_idx = torch.arange(mask_pred.size(0), device=mask_pred.device)
        mask_loss = F.binary_cross_entropy_with_logits(mask_pred[pos_idx, label], mask_target)
        return mask_loss

    def get_mask(self, mask_pred, boxes):
        mask_pred = mask_pred.sigmoid()
        labels = boxes.getlabel()
        index = torch.arange(mask_pred.size(0), device=mask_pred.device)
        mask_pred = mask_pred[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_pred = mask_pred.split(boxes_per_image, dim=0)
        return mask_pred

    def prepare_target(self, proposals, pos_idx, target):
        mask_size = self.mask_size
        mask_targets = []
        for proposal,idx,gt_masks in zip(proposals,pos_idx,target):
            num_pos = proposal.size(0)
            mask_targets_per_img = []
            if num_pos > 0:
                for i in range(num_pos):
                    gt_mask = gt_masks[idx[i]]
                    bbox = proposal[i, :]
                    # crop and resize mask
                    mask = self.maskop(gt_mask,bbox,mask_size)
                    mask_targets_per_img.append(mask)
                mask_targets_per_img = torch.cat(mask_targets_per_img,dim=0)
            else:
                mask_targets_per_img = torch.empty(0,dtype=torch.float32, device=proposal.device)
            mask_targets.append(mask_targets_per_img)
        return mask_targets
