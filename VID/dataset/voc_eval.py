import numpy as np

__all__ = ['voceval']


def bbox_iou_numpy(box1, box2):
    """
    Compute IoU using numpy

    Returns:
        iou n1*n2
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


def _computeAP(recall, precision, use_07_metric=False):
    """Compute AP with recall and precision"""
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
        return ap
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    # make sure the precision is 单调递减
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voceval(n_cls, dets, annos, ignore_thresh=.5, conf=.5):
    """
    Compute mAP and AP for each class

    Arguments:
        n_cls (int)
        dets (list)
    """
    assert len(dets) == len(annos)
    ap = np.zeros(n_cls, dtype=np.double)
    p = np.zeros(n_cls, dtype=np.double)
    r = np.zeros(n_cls, dtype=np.double)
    for label in range(n_cls):
        true_positives = []
        scores = []
        num_annotations = 0

        for detections, annotations in zip(dets, annos):
            if detections is None: continue
            detections = detections[detections[:, -1] == label]
            if len(annotations): annotations = annotations[annotations[:, 4] == label]

            num_annotations += len(annotations)  # .shape[0]
            detected_annotations = []

            if len(detections) == 0: continue
            if len(annotations) == 0:
                for j in range(len(detections)):
                    scores.append(detections[j, -2])
                    true_positives.append(0)
                continue

            # The det whose iou is the largest seems the true positive
            overlaps = bbox_iou_numpy(detections[:, :4], annotations[:, :4])
            assigned_dets = np.argmax(overlaps, axis=0)

            for j in range(len(detections)):
                scores.append(detections[j, -2])

                assigned_det = assigned_dets[j]
                max_overlap = overlaps[assigned_det, j]

                if max_overlap >= ignore_thresh and assigned_det not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_det)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            ap[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score 降序
        indices = np.argsort(-np.array(scores))
        posmask = np.array(scores)[indices] >= conf
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        r[label] = np.sum(true_positives[posmask]) / num_annotations
        p[label] = np.sum(true_positives[posmask]) / np.maximum(
            np.sum(true_positives[posmask] + false_positives[posmask]),
            np.finfo(np.float64).eps)

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _computeAP(recall, precision)
        ap[label] = average_precision

    return ap, p, r
