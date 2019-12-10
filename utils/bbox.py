import numpy as np
import cv2


def AnnoToNumpy(annos, iddict, width, height):
    """Convert a coco annotations into an array

    Note:
        Return xywh type limited in [0,1]
    """
    if not len(annos):
        return []
    r = np.zeros((len(annos), 5))
    for i, ann in enumerate(annos):
        r[i, 0] = iddict[ann['category_id']]
        bbox = ann['bbox']
        r[i, 1] = (bbox[0] + bbox[2] / 2) / width
        r[i, 2] = (bbox[1] + bbox[3] / 2) / height
        r[i, 3] = bbox[2] / width
        r[i, 4] = bbox[3] / height
    return r


def label_to_box(labels, size):
    """Convert ground truth to box"""
    if len(labels) == 0: return labels
    re = np.zeros_like(labels, dtype=np.float32)
    labels[:, 1:5:2] *= size[0]
    labels[:, 2:5:2] *= size[1]
    re[:, 0] = labels[:, 1] - labels[:, 3] / 2
    re[:, 1] = labels[:, 2] - labels[:, 4] / 2
    re[:, 2] = labels[:, 1] + labels[:, 3] / 2
    re[:, 3] = labels[:, 2] + labels[:, 4] / 2
    re[:, 4] = labels[:, 0]
    return re


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(poly[::2])
        x1 = max(poly[::2])
        y0 = min(poly[1::2])
        y1 = max(poly[1::2])
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format.

    Note: w=x2-x1; h=y2-y1
    """
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1
        h = xyxy[3] - y1
        return (x1, y1, w, h)
    else:
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))


def pixelCount(polys, box):
    """
    Count pixel in poly

    Arguments:
        polys
        box : x1y1wh type
    """
    im = np.zeros((int(box[3]), int(box[2])), dtype=np.uint8)
    polyscp = np.copy(polys).astype(np.int)
    polyscp[:, 0] -= int(box[0])
    polyscp[:, 1] -= int(box[1])
    cv2.fillPoly(im, polyscp[np.newaxis], 255)
    return int((im == 255).sum())


def reorg(dets, img_size, input_size, keep_ratio=True):
    """
    Affine predict box to original image

    Arguments:
        dets (array)
        img_size (tuple)
        input_size (tuple)
    """
    if dets is None or len(dets) == 0: return dets
    w, h = img_size
    net_w, net_h = input_size
    scale = min(net_w / w, net_h / h)
    if keep_ratio:
        pad_w = (net_w - w * scale) / 2
        pad_h = (net_h - h * scale) / 2
        dets[:, 0:4:2] -= pad_w
        dets[:, 1:4:2] -= pad_h
    dets[:, :4] /= scale
    return dets


def reorg_ann(anns, img_size, input_size, keep_ratio=True):
    """
    Affine predict box to original image

    Arguments:
        dets (array)
        img_size (tuple)
        input_size (tuple)
    """
    if anns is None or len(anns) == 0: return anns
    w, h = img_size
    net_w, net_h = input_size
    scale = min(net_w / w, net_h / h)
    pad_w = (net_w - w * scale) / 2
    pad_h = (net_h - h * scale) / 2
    x1 = net_w * (anns[:, 1] - anns[:, 3] / 2)
    y1 = net_h * (anns[:, 2] - anns[:, 4] / 2)
    x2 = net_w * (anns[:, 1] + anns[:, 3] / 2)
    y2 = net_h * (anns[:, 2] + anns[:, 4] / 2)
    anns[:, 1] = x1
    anns[:, 2] = y1
    anns[:, 3] = x2
    anns[:, 4] = y2
    if keep_ratio:
        anns[:, 1:5:2] -= pad_w
        anns[:, 2:5:2] -= pad_h
    anns[:, 1:] /= scale
    return anns

def bbox_iou_numpy(box1, box2):
    """
    Arguments:
        box1: x1y1x2y2
        box2: x1y1x2y2
    """
    iw = np.maximum(np.minimum(np.reshape(box1[:, 2], (1, -1)), np.reshape(box2[:, 2], (-1, 1)))
                    - np.maximum(np.reshape(box1[:, 0], (1, -1)), np.reshape(box2[:, 0], (-1, 1))), 0)
    ih = np.maximum(np.minimum(np.reshape(box1[:, 3], (1, -1)), np.reshape(box2[:, 3], (-1, 1)))
                    - np.maximum(np.reshape(box1[:, 1], (1, -1)), np.reshape(box2[:, 1], (-1, 1))), 0)
    intersection = iw * ih
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    ua = np.maximum(np.reshape(area1, (1, -1)) + np.reshape(area2, (-1, 1)) - intersection, np.finfo(float).eps)

    return intersection / ua
