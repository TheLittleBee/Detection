from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import os.path
from pycocotools.coco import COCO
from utils.bbox import AnnoToNumpy
from .coco_eval import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class JsonDataset(Dataset):
    with_bg = False

    def __init__(self, annofile, img_dir, transform=None, city=False):
        self.coco = COCO(annofile)
        self.catid_continue = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.label2catid = {v: k for k, v in self.catid_continue.items()}
        imgIds = self.coco.getImgIds()
        self.imgIds = []
        # remove empty annotation
        for imgid in imgIds:
            imgdict = self.coco.loadImgs(imgid)[0]
            annIds = self.coco.getAnnIds(imgIds=imgdict['id'], iscrowd=False)
            anno = self.coco.loadAnns(annIds)
            if len(anno): self.imgIds.append(imgid)
        self.idx2id = {k: v for k, v in enumerate(self.imgIds)}
        self.img_dir = img_dir
        self.transform = transform
        self.city = city

    def __getitem__(self, index):
        imgdict = self.coco.loadImgs(self.imgIds[index])[0]
        annIds = self.coco.getAnnIds(imgIds=imgdict['id'], iscrowd=False)
        anno = self.coco.loadAnns(annIds)
        if self.city:
            img_path = os.path.join(self.img_dir + imgdict['file_name'].split('_')[0], imgdict['file_name'])
        else:
            img_path = self.img_dir + imgdict['file_name']
        img = Image.open(img_path)
        labels = AnnoToNumpy(anno, self.catid_continue, imgdict['width'], imgdict['height'])
        if self.with_bg:
            labels[:, 0] += 1

        sample = {'img': img, 'label': labels}
        if self.transform:
            sample = self.transform(sample)
        sample['info'] = {'path': img_path, 'id': self.imgIds[index]}
        return sample

    def __len__(self):
        return len(self.imgIds)

    def eval(self, n_cls, dets, ids, input_size, resFile):
        if not os.path.exists(resFile):
            write_coco_resfile(dets, ids, self, input_size, resFile)
        return cocoeval(resFile, self, 'bbox')
