import torch


def bbox_ious(boxes1, boxes2, xywh=False, giou=False):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Arguments:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes
    Note:
        List format: [[x1, y1, x2, y2],...]
    """
    # b1_len = boxes1.size(0)
    # b2_len = boxes2.size(0)
    if xywh:
        boxes1 = xywh2xyxy(boxes1)
        boxes2 = xywh2xyxy(boxes2)
    b1x1, b1y1, b1x2, b1y2 = boxes1.split(1, 1)
    b2x1, b2y1, b2x2, b2y2 = boxes2.split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections
    iou = intersections / unions

    if giou:
        dx = (b1x2.max(b2x2.t()) - b1x1.min(b2x1.t())).clamp(min=0)
        dy = (b1y2.max(b2y2.t()) - b1y1.min(b2y1.t())).clamp(min=0)
        encloses = dx * dy
        return iou - (encloses - unions) / encloses

    return iou


def xywh2xyxy(boxes):
    """Convert boxes from xywh type to xyxy type"""
    x1 = boxes[:, 0] - 0.5 * boxes[:, 2]
    y1 = boxes[:, 1] - 0.5 * boxes[:, 3]
    x2 = boxes[:, 0] + 0.5 * boxes[:, 2]
    y2 = boxes[:, 1] + 0.5 * boxes[:, 3]
    res = torch.stack((x1, y1, x2, y2), 1)
    return res


def concat_box_prediction(classification, regression):
    """
    Concat cls and reg & reshape cls like (N,-1,C) and reg like (N,-1,4)

    Arguments:
        classification (list[Tensor])
        regression (list[Tensor])
    """
    clslist = []
    boxlist = []
    for cls, box in zip(classification, regression):
        N, AxC, H, W = cls.shape
        Ax4 = box.shape[1]
        A = Ax4 // 4
        C = AxC // A
        clslist.append(cls.view(N, A, C, H, W).permute(0, 3, 4, 1, 2).contiguous().view(N, -1, C))
        boxlist.append(box.view(N, A, 4, H, W).permute(0, 3, 4, 1, 2).contiguous().view(N, -1, 4))
    clslist = torch.cat(clslist, dim=1)
    boxlist = torch.cat(boxlist, dim=1)
    return clslist, boxlist


def anchor_decoding(anchor, regression, weight=(10., 10., 5., 5.)):
    """
    Get predict bbox

    Arguments:
        anchor (Tensor(n,4))
        regression (Tensor(n,K*4))
    Note:
        regression should have same length as anchor
    Returns:
        pred (Tensor(n,K*4)): xyxy type
    """
    assert len(anchor) == len(regression)
    widths = anchor[:, 2] - anchor[:, 0]
    heights = anchor[:, 3] - anchor[:, 1]
    ctr_x = anchor[:, 0] + 0.5 * widths
    ctr_y = anchor[:, 1] + 0.5 * heights
    dx = regression[:, 0::4] / weight[0]
    dy = regression[:, 1::4] / weight[1]
    dw = regression[:, 2::4] / weight[2]
    dh = regression[:, 3::4] / weight[3]
    pred_ctr_x = ctr_x[:, None] + dx * widths[:, None]
    pred_ctr_y = ctr_y[:, None] + dy * heights[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]
    predx1 = (pred_ctr_x - pred_w * 0.5).view(-1, 1)
    predy1 = (pred_ctr_y - pred_h * 0.5).view(-1, 1)
    predx2 = (pred_ctr_x + pred_w * 0.5).view(-1, 1)
    predy2 = (pred_ctr_y + pred_h * 0.5).view(-1, 1)
    pred = torch.cat((predx1, predy1, predx2, predy2), 1).view(regression.size(0), -1)
    return pred


def anchor_encoding(anchor, target, weight=(10., 10., 5., 5.)):
    """
    Get regression target

    Arguments:
        anchor (Tensor(n,4))
        target (Tensor(n,4)): same length as anchor
    """
    assert len(anchor) == len(target)
    anchor_width = anchor[:, 2] - anchor[:, 0]
    anchor_height = anchor[:, 3] - anchor[:, 1]
    anchor_ctrx = anchor[:, 0] + 0.5 * anchor_width
    anchor_ctry = anchor[:, 1] + 0.5 * anchor_height
    gt_width = target[:, 2] - target[:, 0]
    gt_height = target[:, 3] - target[:, 1]
    gt_ctrx = target[:, 0] + 0.5 * gt_width
    gt_ctry = target[:, 1] + 0.5 * gt_height
    coord_x = (gt_ctrx - anchor_ctrx) / anchor_width * weight[0]
    coord_y = (gt_ctry - anchor_ctry) / anchor_height * weight[1]
    coord_w = torch.log(gt_width / anchor_width) * weight[2]
    coord_h = torch.log(gt_height / anchor_height) * weight[3]
    return torch.stack((coord_x, coord_y, coord_w, coord_h), -1)


def clip2img(boxes, img_size, remove_empty=True):
    """
    Limit boxes in image size

    Arguments:
        boxes (Tensor(n,>4)): xyxy type
    """
    TO_REMOVE = 1
    boxes[:, 0].clamp_(min=0, max=img_size[0] - TO_REMOVE)
    boxes[:, 1].clamp_(min=0, max=img_size[1] - TO_REMOVE)
    boxes[:, 2].clamp_(min=0, max=img_size[0] - TO_REMOVE)
    boxes[:, 3].clamp_(min=0, max=img_size[1] - TO_REMOVE)
    if remove_empty:
        box = boxes
        keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
        return boxes[keep]
    return boxes


def remove_small(boxes, min_size):
    """Remove small boxes"""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (
            (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxes[keep]
