import torch
import torch.nn.functional as F
from torch import nn

from ...utils.roi_pool import RoIPool
from ...utils.roi_align import RoIAlign


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, proposals):
        """
        Arguments:
            proposals (list[tensor])
        """
        # Compute level ids
        boxlists = torch.cat(proposals, dim=0)
        s = torch.sqrt((boxlists[:, 3] - boxlists[:, 1] + 1) * (boxlists[:, 2] - boxlists[:, 0] + 1))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, type, output_size, scales, sample_ratio=None):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            if type == 'RoIAlign':
                assert sample_ratio is not None
                poolers.append(RoIAlign(output_size, spatial_scale=scale, sample_ratio=sample_ratio))
            else:
                poolers.append(RoIPool(output_size, spatial_scale=scale))
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        if isinstance(output_size, (int)):
            self.output_size = (output_size, output_size)
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        """

        Arguments:
            boxes (list[Tensor(n,5)])
        Returns:
            rois (Tensor(n,5)): batch_idx, box
        """
        concat_boxes = torch.cat(boxes, dim=0)[:, :4]
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat((ids, concat_boxes), dim=1)
        return rois

    def forward(self, x, proposals):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level, low resolution first
            proposals (list[tensor(n,5)]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(proposals)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(proposals)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size

        dtype, device = x[0].dtype, x[0].device
        # result = torch.zeros(
        #     (num_rois, num_channels,) + output_size,
        #     dtype=dtype,
        #     device=device,
        # )
        # for level, (per_level_feature, pooler) in enumerate(zip(x[::-1], self.poolers)):  # high resolution first
        #     idx_in_level = torch.nonzero(levels == level).squeeze(1)
        #     if len(idx_in_level) == 0: continue
        #     rois_per_level = rois[idx_in_level]
        #     result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)
        result = []
        for i, r in enumerate(rois):
            l = levels[i]
            result.append(self.poolers[l](x[-1 - l], r).to(dtype))
        result = torch.cat(result, dim=0)

        return result
