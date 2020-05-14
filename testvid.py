import argparse
import time
import torch
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import os.path as path

from utils.nms_wrapper import getResult
from utils.bbox import reorg
import VID.dataset as dt
from VID.models.yolo import YOLO
from littlenet.utils.bbox import clip2img

valanns = '/home/littlebee/dataset/VID/ILSVRC2015/annotations_val.pkl'
valimg = '/home/littlebee/dataset/VID/ILSVRC2015/Data/VID/val'
classes = ["airplane", "antelope", "bear", "bicycle", "bird", "bus", "car",
           "cattle", "dog", "domestic_cat", "elephant", "fox", "giant_panda",
           "hamster", "horse", "lion", "lizard", "monkey", "motorcycle", "rabbit",
           "red_panda", "sheep", "snake", "squirrel", "tiger", "train", "turtle",
           "watercraft", "whale", "zebra"]


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--weight', dest='weight', type=str, required=True, help='Weight file')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--keepsize', action='store_true')
    parser.add_argument('-v', '--vis', dest='visual', action='store_true', help='Visualize results')
    args = parser.parse_args()
    return args


args = parse_args()
print(args)
cuda = torch.cuda.is_available()

print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + 'Starting inference')

if args.visual:
    from utils.visualize import Debugger

    debuger = Debugger(classes)

with open(args.model, 'r') as fp:
    cfg = yaml.load(fp)
model_cfg = cfg['model']
assert model_cfg.pop('type') == 'YOLO'
bg = model_cfg.pop('background')
model_cfg['weights'] = args.weight
model_cfg['head']['num_classes'] = 31 if bg else 30
net = YOLO(**model_cfg)
if cuda: net.cuda()
net.eval()

transform = tf.Compose([dt.Resize_Pad(416), dt.ToTensor()])
if args.keepsize:
    transform = tf.Compose([dt.ToTensor()])
valdata = dt.VIDDataset(valanns, valimg, 1, 1, transform)
valloader = DataLoader(
    valdata,
    batch_size=1,
    num_workers=0,
    # pin_memory=pin_mem if cuda else False,
    collate_fn=dt.list_collate,
)

all_det = {}
pid = None
tot_time = 0

for _, sample in enumerate(tqdm(valloader)):
    img, target = sample['img'], sample['label']
    id = sample['info'][0]['id']
    if id != pid:
        net.clear()
        all_det[id] = []
    if cuda: img = img.cuda()
    start = time.time()
    with torch.no_grad():
        out = net(img, target)
        det = [clip2img(d, valdata.img_size[id] if args.keepsize else (416, 416), remove_empty=True) for d in out[0]]
        det = getResult(det, 30, 100, bg, args.conf, args.nms)
    tot_time += time.time() - start
    pid = id

    det = det[0].cpu().numpy() if det[0] is not None else None
    if not args.keepsize: det = reorg(det, valdata.img_size[id], (416, 416))
    if args.visual:
        oup = net.meta['rnn'][0].norm(p=2,dim=0,keepdim=True).cpu().numpy()
        oup_c = debuger.gen_colormap(oup)
        back = img[0].squeeze(0).detach().cpu().numpy() * 255
        debuger.add_blend_img(np.transpose(back, (1, 2, 0)), oup_c, img_id='rnn')
        # debuger.show_img(img_id='rnn', pause=1)
        img_ori = cv2.imread(path.join(valimg, id, '{:0>6}.JPEG'.format(len(all_det[id]))))
        debuger.add_2d_detection(img_ori, det, thresh=0.1, img_id='det')
        # debuger.show_img(img_id='det', pause=1)
        debuger.show_all_imgs()
    all_det[id].append(det)

print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + 'Now computing AP')
print('{:>12}: {:.4f}s'.format('Avg time', tot_time / len(valloader)))

res = valdata.eval(30, all_det)
for k, v in res.items():
    if isinstance(k, int): k = classes[k]
    if isinstance(v, (list, tuple)):
        str = '{:>12}: '.format(k)
        for vl in v:
            str = str + '{:.4f}, '.format(vl)
        print(str)
    else:
        print('{:>12}: {:.4f}'.format(k, v))
