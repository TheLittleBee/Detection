from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.bbox import *

__all__ = ['cocoeval', 'write_coco_resfile']


def cocoeval(resFile, dataset, annType):
    """
    Evaluate result using coco tools

    Arguments:
        resFile (file)
        dataset (JsonDataset): assert JsonDataset type
        annType : assert in ['segm','bbox','keypoints']
    """
    # TODO each class's AP
    res = {'AP': 0, 'AP50': 0, 'AP75': 0, 'APs': 0, 'APm': 0, 'APl': 0, 'AR1': 0, 'AR10': 0, 'AR100': 0, 'ARs': 0,
           'ARm': 0, 'ARl': 0}
    cocoGt = dataset.coco
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    for i, k in enumerate(res):
        res[k] = cocoEval.stats[i]
    return res


def write_coco_resfile(dets, ids, dataset, input_size, resfile):
    """
    Write result to json file with coco type

    Arguments:
        dets (list)
        ids (list)
    """
    coco_results = []
    for image_id, prediction in zip(ids, dets):
        if len(prediction) == 0:
            continue

        img_info = dataset.coco.imgs[image_id]
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.cpu().numpy()
        prediction = reorg(prediction, (image_width, image_height), input_size)
        boxes = xyxy_to_xywh(prediction).tolist()
        scores = prediction[:, 4].tolist()
        labels = prediction[:, -1]
        if dataset.with_bg:
            labels = labels - 1

        mapped_labels = [dataset.label2catid[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    import json
    with open(resfile, "w") as f:
        json.dump(coco_results, f)
