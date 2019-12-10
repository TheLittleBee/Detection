import torch
import numpy as np
import time
import cv2
import argparse
import onnxruntime as ort
import onnx

from littlenet import models
from utils.nms_wrapper import *
from utils.config import yml_parse

INPUT_SIZE = (416, 416)
LABELS = ["car", "person", "tank"]
MEAN = (102.9801, 115.9465, 122.7717)
STD = (1.0, 1.0, 1.0)
# MEAN = (104., 117., 123.)
# MEAN = (103.53, 116.28, 123.675)
# STD = (57.375, 57.12, 58.395)


def parse_args():
    parser = argparse.ArgumentParser(description='Do inference')
    parser.add_argument('-n', '--num', dest='n_cls', type=int, required=True, help='Num classes')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--wt', dest='weight', type=str, required=True, help='Weight file')
    parser.add_argument('--img', dest='image', type=str, required=True, help='Image file')
    parser.add_argument('-d', '--data', dest='data', type=str, default='LAB', help='Dataset name')
    args = parser.parse_args()
    return args


def build_model(args):
    model_cfg = yml_parse(args.model)['model']
    model_type = model_cfg.pop('type')
    bg = model_cfg.pop('background')
    assert model_type in models.__dict__
    model_cfg['weights'] = args.weight
    model_cfg['head']['num_classes'] = args.n_cls
    net = models.__dict__[model_type](**model_cfg)
    net.input_size = INPUT_SIZE
    net.num_classes = args.n_cls
    return net, bg


def img_detect(model, img, bg=False, conf_thresh=0.5, nms_thresh=0.4):
    with torch.no_grad():
        out = model(img)
        # print(out)
        res = getResult(out, model.num_classes, 20, bg, conf_thresh, nms_thresh)
    return res


def preprocess(src, keep_ratio=True, norm=False):
    if not keep_ratio:
        src = cv2.resize(src, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    else:
        h, w = src.shape[:2]
        net_w, net_h = INPUT_SIZE
        if w != net_w or h != net_h:
            scale = min(net_h / h, net_w / w)
            if scale != 1:
                src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                h, w = src.shape[:2]
            if w != net_w or h != net_h:
                pad_w = (net_w - w) / 2
                pad_h = (net_h - h) / 2
                src = cv2.copyMakeBorder(src, int(pad_h), int(pad_h + .5), int(pad_w), int(pad_w + .5),
                                         cv2.BORDER_CONSTANT,
                                         value=(128,) * src.shape[2])
    if norm:
        src = (src - MEAN) / STD
    else:
        src = src / 255.
    img = torch.from_numpy(np.transpose(src, (2, 0, 1))[np.newaxis]).float()
    return img


def visualize(img, dets, labels, file, t=1., keep_ratio=True):
    if dets is None:
        print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + 'NO OBJ!')
    else:
        h, w = img.shape[:2]
        net_w, net_h = INPUT_SIZE
        if keep_ratio:
            scale = min(net_w / w, net_h / h)
            pad_w = (net_w - w * scale) / 2
            pad_h = (net_h - h * scale) / 2
            dets[:, 0:4:2] -= pad_w
            dets[:, 1:4:2] -= pad_h
            dets[:, :4] /= scale
        else:
            dets[:, 0:4:2] *= w / net_w
            dets[:, 1:4:2] *= h / net_h
        for x1, y1, x2, y2, conf, cls in dets:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            txt_loc = (int(max(x1, 5)), int(max(y1 - 5, 20)))
            txt = '{} {:.2f}'.format(labels[int(cls)], conf)
            cv2.putText(img, txt, txt_loc, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
        txt = 'FPS: %.2f' % (1. / t)
        cv2.putText(img, txt, (w - 150, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
        cv2.imwrite(file, img)
        cv2.imshow('result', img)
        cv2.waitKey()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.data:
        LABELS = yml_parse('cfg/dataset.yml')[args.data]['labels']
    model, bg = build_model(args)
    # print(model.state_dict())
    # model.save_weights('weights/cascade_rcnn_resnet50_fpn_COCO.pth')
    img = cv2.imread(args.image)
    input_img = preprocess(img.copy(), norm=False, keep_ratio=False)
    cuda = torch.cuda.is_available()
    model.eval()
    if cuda:
        model.cuda()
        input_img = input_img.cuda()
    tic = time.time()
    # md = onnx.load('result/yolov3_mobilenetv2_VOC_export.onnx')
    # onnx.checker.check_model(md)
    # print(input_img.shape)
    # ort_session = ort.InferenceSession('result/yolov3_mobilenetv2_VOC_export.onnx')
    # out = ort_session.run(None, {"0": input_img.detach().numpy()})
    # print(out)
    # model.exporting = True
    # img_detect(model, input_img, bg)
    for i in range(10):
        res = img_detect(model, input_img, bg, 0.5, 0.4)
    tot = time.time() - tic
    resultfile = 'result/' + args.image.split('/')[-1]
    visualize(img, res[0], LABELS, resultfile, tot / 10.,keep_ratio=False)
