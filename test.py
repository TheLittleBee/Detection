import argparse
import time
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from utils.config import HyperParams
from engine._test import TestEngine
from utils.nms_wrapper import getResult
from utils.bbox import reorg, reorg_ann


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('-d', '--data', dest='data', type=str, default='CITY', help='Dataset to use')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--weight', dest='weight', type=str, required=True, help='Weight file')
    parser.add_argument('-v', '--vis', dest='visual', action='store_true', help='Visualize results')
    args = parser.parse_args()
    return args


args = parse_args()
print(args)

hyper = HyperParams(args, 2)
engine = TestEngine(hyper)
print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + 'Starting inference')
all_det = []
all_ids = []
tot_time = 0
if args.visual:
    from utils.visualize import Debugger

    debuger = Debugger(engine.classes)
for _, sample in enumerate(tqdm(engine.dataloader)):
    img, target = sample['img'], sample['label']
    if engine.cuda: img = img.cuda()
    start = time.time()
    with torch.no_grad():
        out = engine.net(img, target)
        det = getResult(out, engine.n_cls, engine.n_det, engine.bg, engine.conf_thresh, engine.nms_thresh)
    tot_time += time.time() - start

    if engine.cuda:
        det = [d.cpu() if d is not None else None for d in det]
    for i in range(len(det)):
        if args.visual:
            img_ori = cv2.imread(sample['info'][i]['path'])
            det_ori = reorg(np.array(det[i]) if det[i] is not None else None, img_ori.shape[:2][::-1],
                            engine.input_size)
            anno = reorg_ann(target[i], img_ori.shape[:2][::-1], engine.input_size)
            for k, v in engine.net.meta.items():
                oup = v[i].cpu().numpy()
                oup_c = debuger.gen_colormap(oup)
                back = img[i].squeeze(0).detach().cpu().numpy() * 255
                debuger.add_blend_img(np.transpose(back, (1, 2, 0)), oup_c, img_id=k)
                debuger.show_img(img_id=k)
            debuger.add_2d_detection(img_ori, det_ori, anno, thresh=0.1, img_id=sample['info'][i]['id'])
            debuger.show_img(img_id=sample['info'][i]['id'])
        all_det.append(det[i])
        info = sample['info'][i]
        all_ids.append(info['id'])
        # pdet = []
        # if det[i] is not None and len(det[i]):
        #     pred_boxes = det[i][:, :5].cpu().numpy()
        #     pred_labels = det[i][:, -1].cpu().numpy()
        #
        #     for j in range(engine.n_cls):
        #         pdet.append(pred_boxes[pred_labels == j])
        # else:
        #     for j in range(engine.n_cls): pdet.append([])
        # all_det.append(pdet)
        # panno = []
        # if target[i] is not None and len(target[i]):
        #     anno_labels = target[i][:, 0]
        #     anno_boxes = np.empty_like(target[i][:, 1:])
        #
        #     anno_boxes[:, 0] = target[i][:, 1] - target[i][:, 3] / 2
        #     anno_boxes[:, 1] = target[i][:, 2] - target[i][:, 4] / 2
        #     anno_boxes[:, 2] = target[i][:, 1] + target[i][:, 3] / 2
        #     anno_boxes[:, 3] = target[i][:, 2] + target[i][:, 4] / 2
        #     anno_boxes[:, 0:4:2] *= engine.input_size[0]
        #     anno_boxes[:, 1:4:2] *= engine.input_size[1]
        #
        #     for j in range(engine.n_cls):
        #         panno.append(anno_boxes[anno_labels == j])
        # else:
        #     for j in range(engine.n_cls): panno.append([])
        # all_anno.append(panno)

print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + 'Now computing AP')
print('{:>12}: {:.4f}s'.format('Avg time', tot_time / len(engine.dataloader)))
# AP = computeAP(engine.n_cls, all_det, all_anno, engine.ignore_thresh)
#
# for i in range(engine.n_cls):
#     print('+ Class {} - AP: {}'.format(engine.classes[i], AP[i]))
# mAP = np.mean(AP)
# print('mAP: {}'.format(mAP))
res = engine.dataset.eval(engine.n_cls, all_det, all_ids, engine.input_size,
                          'result/%s_%s.json' % (engine.model_name, args.data))
for k, v in res.items():
    if isinstance(k, int): k = engine.classes[k]
    if isinstance(v, (list, tuple)):
        str = '{:>12}: '.format(k)
        for vl in v:
            str = str + '{:.4f}, '.format(vl)
        print(str)
    else:
        print('{:>12}: {:.4f}'.format(k, v))
