import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.bbox import *
from .utils import *
from ..utils.nms.nms_wrapper import *
from .anchor_based import AnchorLoss


class RPNLoss(AnchorLoss):
    def __init__(self, anchors, input_size, min_size, train_pre_n, train_post_n, test_pre_n, test_post_n,
                 num, pos_fraction, pos_iou_thr, neg_iou_thr, nms_thresh=0.45, coder_w=(1., 1., 1., 1.)):
        """

        """
        super(RPNLoss, self).__init__(anchors, input_size, pos_iou_thr, neg_iou_thr, coder_w)
        self.min_size = min_size
        self.train_pre_n = train_pre_n
        self.train_post_n = train_post_n
        self.pre_n = test_pre_n
        self.post_n = test_post_n
        self.num = num
        self.pos_fraction = pos_fraction
        self.nms_thresh = nms_thresh
        self.info = {'recall50': 0, 'recall75': 0}

    def forward(self, output, target=None):
        """

        Arguments:
            classification (list[Tensor]): low resolution first
            regression (list[Tensor]): low resolution first
            target (list[array(n,5)])
        """
        assert isinstance(output, (list, tuple)) and len(output) == 2
        classification, regression = output
        device = classification[0].device
        if target is not None:
            self.pre_n = self.train_pre_n
            self.post_n = self.train_post_n
        self.anchors = [anchors.to(device) for anchors in self.anchors]
        sampled_boxes = []  # list[list[tensor]_NB]_NF
        num_levels = len(classification)
        for a, c, r in zip(self.anchors, classification, regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, c, r))

        boxlists = list(zip(*sampled_boxes))  # list[list[tensor]_NF]_NB
        proposals = [torch.cat(boxlist, dim=0) for boxlist in boxlists]

        if num_levels > 1:
            proposals = self.select_over_all_levels(proposals)
        if target is None:
            return proposals, ()
        self.anchors = torch.cat(self.anchors, dim=0)
        labels, coords = self.match_targets(target)
        pos_mask, neg_mask = self.resample(labels)
        labels = labels[..., None].float()
        for k in self.info:
            self.info[k] /= len(proposals)
        sampled_pos_mask = pos_mask > 0
        sampled_neg_mask = neg_mask > 0

        sampled_mask = sampled_pos_mask | sampled_neg_mask
        classification, regression = concat_box_prediction(classification, regression)
        coord_loss = smooth_l1_loss(regression[sampled_pos_mask], coords[sampled_pos_mask], beta=1. / 9,
                                    size_average=False)
        cls_loss = F.binary_cross_entropy_with_logits(classification[sampled_mask], labels[sampled_mask])
        return proposals, cls_loss, coord_loss / sampled_mask.nonzero().size(0)

    def forward_for_single_feature_map(self, anchors, classification, regression):
        """

        Arguments:
            anchors (Tensor(n,4))
            classification (Tensor)
            regression (Tensor)
        Returns:
            result (list[Tensor(n,5)])
        """
        device = classification.device
        N, A, H, W = classification.shape

        # put in the same format as anchors
        classification = classification.permute(0, 2, 3, 1).contiguous().view(N, -1)
        classification = classification.sigmoid()

        regression = regression.view(N, A, 4, H, W).permute(0, 3, 4, 1, 2).contiguous().view(N, -1, 4)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_n, num_anchors)
        classification, topk_idx = classification.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        regression = regression[batch_idx, topk_idx]

        image_size = self.input_size
        c_anchors = anchors.unsqueeze(0).expand((N,) + anchors.size())[batch_idx, topk_idx]

        proposals = anchor_decoding(c_anchors.view(-1, 4), regression.view(-1, 4), self.coder_w)

        proposals = proposals.view(N, -1, 4)

        result = []
        for proposal, score in zip(proposals, classification):
            boxlist = torch.cat((proposal, score[:, None]), -1)
            boxlist = clip2img(boxlist, image_size)
            boxlist = remove_small(boxlist, self.min_size)
            _, keep = nms(
                boxlist,
                self.nms_thresh,
            )
            result.append(boxlist[keep[:self.post_n]])
        return result

    def select_over_all_levels(self, boxlists):
        """

        Arguments:
            boxlists (list[Tensor(n,5)])
        """
        for i in range(len(boxlists)):
            objectness = boxlists[i][:, 4]
            post_nms_top_n = min(self.post_n, len(objectness))
            _, inds_sorted = torch.topk(
                objectness, post_nms_top_n, dim=0, sorted=True
            )
            boxlists[i] = boxlists[i][inds_sorted]
        return boxlists

    def resample(self, labels):
        """

        """
        pos_mask = torch.zeros_like(labels, dtype=torch.uint8, device=labels.device)
        neg_mask = torch.zeros_like(labels, dtype=torch.uint8, device=labels.device)
        for i in range(labels.size(0)):
            positive = torch.nonzero(labels[i] == 1).squeeze(1)
            negative = torch.nonzero(labels[i] == 0).squeeze(1)

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
            pos_mask[i][pos_idx_per_image] = 1
            neg_mask[i][neg_idx_per_image] = 1

        return pos_mask, neg_mask
