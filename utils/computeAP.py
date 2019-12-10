import numpy as np
from .bbox import bbox_iou_numpy


def _computeAP(recall, precision, use_07_metric=False):
    '''矩形和来计算面积'''
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


def computeAP(n_cls, det, anno, ignore_thresh=.5):
    assert len(det) == len(anno)
    ap = np.zeros(n_cls, dtype=np.double)
    for label in range(n_cls):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in range(len(anno)):
            detections = det[i][label]
            annotations = anno[i][label]

            num_annotations += len(annotations)  # .shape[0]
            detected_annotations = []

            if len(detections) == 0: continue
            if len(annotations) == 0:
                for j in range(len(detections)):
                    scores.append(detections[j, 4])
                    true_positives.append(0)
                continue

            # The det whose iou is the largest seems the true positive
            overlaps = bbox_iou_numpy(detections[:, :4], annotations)
            assigned_dets = np.argmax(overlaps, axis=0)

            for j in range(len(detections)):
                scores.append(detections[j, 4])

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
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _computeAP(recall, precision)
        ap[label] = average_precision

    return ap
