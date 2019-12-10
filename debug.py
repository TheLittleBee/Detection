import dataset
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import Compose
import cv2
import numpy as np
from littlenet import models
import torch
import random
# from utils.nms_wrapper import getResult


# labels=["person", "Truck", "car", "Cyclist", "Misc", "Van", "Tram", "Person_sitting"]
# data = dataset.VOCDataset('data/VOC2007/annotations_train2007.pkl', 'data/VOC2007/JPEGImages/',transform=Compose([
#     dataset.SSDCrop(),
#     # dataset.RCM((416,416),0.3),
#     # dataset.RandomCrop(0.3),
#     # dataset.Resize_Pad((416,416))
# ]))
# data = dataset.VOCDataset('data/KITTI/annotations_train.pkl', 'data/KITTI/JPEGImages/', transform=Compose([
#     # dataset.Focus((640,320)),
#     # dataset.RandomCrop(416, .8),
#     # dataset.HFlip(),
#     # dataset.ColorJitter(1.5, 1.5, .1),
#     dataset.Resize_Pad((1024,512)),
# ]))
# data = dataset.JsonDataset('data/Cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
#                            'data/Cityscapes/leftImg8bit/val/',transform=Compose([
#         dataset.SSDCrop(),
#         # dataset.Focus((640,320)),
#         # dataset.HFlip(),
#         # dataset.Resize_Pad((640,320)),
#     ]),city=True)
# #
# ns=np.zeros(8)
# # randidx = np.random.randint(0, len(data), 100)
# # for i in randidx:
# boxes = []
# count = 0
# smallc = 0
# for i in range(len(data)):
#     s = data[i]
#     img = np.array(s['img'])
#     imgh, imgw = img.shape[:2]
#     # sm = False
#     for c, x, y, w, h in s['label']:
#         if w == 0 or h == 0: continue
#         box_h = int(h * imgh)
#         box_w = int(w * imgw)
#         y1 = int((y - h / 2) * imgh)
#         x1 = int((x - w / 2) * imgw)
#         cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), (255, 0, 0), 3)
#     #     count += 1
#     #     if box_w<32 and box_h<32:
#     #         boxes.append([box_w,box_h])
#     #         sm = True
#     # if sm: smallc += 1
#     plt.imshow(img)
#     plt.show()
#         ns[int(c)]+=1
# for i in range(8):
#     print('%s : %d'%(labels[i],ns[i]))
# print('small obj:', len(boxes))
# print('ratio:',count)
# print('img has small obj:',smallc)
# boxes = np.array(boxes)
# print('avg:',boxes.mean(axis=0))
# md = models.YOLOv3(80, 'weights/yolov3.pth')
# print(md.state_dict())
# torch.save(md, 'weights/yolov3-COCO.pth')

# md = models.RetinaNet(10,blocks=50)
# md.train()
# X = torch.zeros(2,3,416,416)
# target=[np.array([[2,0.5027343750000001,0.5484375,0.9304687500000001,0.44479166666666664],
# [1,0.6453125000000001,0.3734375,0.051562500000000004,0.096875]]),
# np.array([[2,0.5127504553734062, 0.5126582278481012, 0.8142076502732241, 0.6835443037974683],
# [1, 0.5255009107468124, 0.1518987341772152, 0.08925318761384335, 0.23417721518987342]])]
# loss = md(X,target)
# print(loss)

# with open('/home/littlebee/code/pytorch/Detection/cfg/yolov3.yml', 'r') as fp:
#     cfg = yaml.load(fp)['model']
# cfg['weights'] = '/home/littlebee/code/pytorch/Detection/weights/yolov3.weights'
# cfg['head']['num_classes']=80
# del cfg['type']
# md = models.YOLO(**cfg)
# md.eval()
# img = cv2.imread('/home/littlebee/code/pytorch/Detection/data/dog.jpg')
# img = cv2.resize(img, (416, 640)) / 255.
# x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
# x = x.expand((2,)+x.size())
# with torch.no_grad():
#     out = md(x)
#     res = getResult(out, 80)
# print(res)

# with open('cfg/retinanet_resnet50.yml') as fp:
#     cfg = yaml.load(fp)['model']
# del cfg['type']
# cfg['head']['num_classes'] = 81
# md = models.RetinaNet(**cfg)
# md.input_size = (416, 640)
# md.init_weights()
# if torch.cuda.is_available():
#     md.cuda()
#     x = x.cuda()
# md.train()
# out = md(x,target)
# md.load_weights('backup/Yolov3.weight')
# state = torch.load('weights/imagenet_resnet101.pth',map_location='cpu')
# print(state)
# md.save_weights('weights/yolov3_VOC.pth')
# for k in state['state_dict'].keys():
#     print(k)
# fp = open('weights/yolov3-tiny.weights','wb')
# for m in md.modules():
#     if isinstance(m,torch.nn.Conv2d):
#         m.weight.data.cpu().numpy().tofile(fp)
#         if m.bias is not None:
#             m.bias.data.cpu().numpy().tofile(fp)
#     if isinstance(m,torch.nn.BatchNorm2d):
#         m.weight.data.cpu().numpy().tofile(fp)
#         m.bias.data.cpu().numpy().tofile(fp)
#         m.running_mean.data.cpu().numpy().tofile(fp)
#         m.running_var.data.cpu().numpy().tofile(fp)
# # for k,v in state['state_dict'].items():
# #     if 'num_batches_tracked' in k: continue
# #     v.data.cpu().numpy().tofile(fp)
# fp.close()