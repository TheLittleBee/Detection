import torch

from littlenet.utils.nms.nms_wrapper import nms, soft_nms

__all__ = ['getResult']


def getResult(prediction, num_classes, num_detect, with_bg=False, conf_thresh=0.05, nms_thresh=0.4):
    """
    Post progress for detection, nms with class

    Arguments:
        prediction (list[Tensor]): (x1,y1,x2,y2,score,idx)
        num_classes (int)
        num_detect (int)
    Returns:
        output (list)
    """
    output = []
    for img_pred in prediction:
        conf_mask = (img_pred[:, 4] >= conf_thresh).squeeze()
        img_pred = img_pred[conf_mask]
        if not img_pred.size(0):
            output.append(None)
            continue
        if img_pred.shape[1] > 6:
            cls_conf, cls_pred = torch.max(img_pred[:, 5:5 + num_classes], 1, keepdim=True)
            unique_labels = cls_pred.unique()
            detection = torch.cat((img_pred[:, :4], cls_conf, cls_pred.float()), 1)
        else:
            detection = img_pred
            unique_labels = img_pred[:, 5].unique()
        img_output = []
        for c in unique_labels:
            if with_bg and c == 0: continue
            detection_c = detection[detection[:, -1] == c.float()]
            d, _ = nms(detection_c, nms_thresh)
            img_output.append(d)
        if len(img_output) == 0:
            output.append(None)
            continue
        det = torch.cat(img_output, dim=0)
        # one box maybe has more than one class label
        # det, _ = nms(det, 0.7)
        if len(det) > num_detect:
            score = det[:, 4]
            _, keep = torch.topk(score, num_detect)
            det = det[keep]
        if with_bg: det[:, 5] -= 1
        output.append(det)

    return output
