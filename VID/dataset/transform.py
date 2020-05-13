import torchvision.transforms.functional as TF
import torch
from torch.utils.data.dataloader import default_collate
import random
import numpy as np
from PIL import Image, ImageOps

__all__ = ['Resize_Pad', 'HFlip', 'ColorJitter', 'ToTensor', 'list_collate']


class Resize_Pad():
    def __init__(self, size, keep_ratio=True):
        '''size: w,h'''
        assert isinstance(size, (int, list, tuple))
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, sample):
        if isinstance(sample['img'], list):
            images = sample['img']
            labels = sample['label']
        else:
            images = [sample['img']]
            labels = [sample['label']]
        w, h = images[0].size
        net_w, net_h = self.size
        if self.keep_ratio:
            if w == net_w and h == net_h:
                return {'img': images, 'label': labels}
            scale = min(net_h / h, net_w / w)
            if scale != 1:
                images = [img.resize((int(w * scale), int(h * scale)), Image.NEAREST) for img in images]
                w, h = images[0].size
            if w == net_w and h == net_h:
                return {'img': images, 'label': labels}
            pad_w = (net_w - w) / 2
            pad_h = (net_h - h) / 2
            pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
            images = [ImageOps.expand(img, border=pad, fill=(127, 127, 127)) for img in images]
            for label in labels:
                if len(label) == 0: continue
                x1 = w * (label[:, 1] - label[:, 3] / 2)
                y1 = h * (label[:, 2] - label[:, 4] / 2)
                x2 = w * (label[:, 1] + label[:, 3] / 2)
                y2 = h * (label[:, 2] + label[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[1]
                x2 += pad[0]
                y2 += pad[1]
                # Calculate ratios from coordinates
                label[:, 1] = ((x1 + x2) / 2) / net_w
                label[:, 2] = ((y1 + y2) / 2) / net_h
                label[:, 3] *= w / net_w
                label[:, 4] *= h / net_h
        else:
            images = [img.resize(self.size, Image.NEAREST) for img in images]

        return {'img': images, 'label': labels}


class HFlip():
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, sample):
        if isinstance(sample['img'], list):
            images = sample['img']
            labels = sample['label']
        else:
            images = [sample['img']]
            labels = [sample['label']]
        if random.random() < self.p:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            w, h = images[0].size
            for label in labels:
                if len(label) == 0: continue
                label[:, 1] = ((1 - label[:, 1]) * w - 1) / w

        return {'img': images, 'label': labels}


class ColorJitter():
    def __init__(self, contrast=1., saturation=1., hue=0.):
        self.contrast = [2 - contrast, contrast]
        self.saturation = [2 - saturation, saturation]
        self.hue = [-hue, hue]

    def __call__(self, sample):
        if isinstance(sample['img'], list):
            images = sample['img']
        else:
            images = [sample['img']]
        if self.contrast[0] != self.contrast[1]:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            images = [TF.adjust_contrast(img, contrast_factor) for img in images]
        if self.saturation[0] != self.saturation[1]:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            images = [TF.adjust_saturation(img, saturation_factor) for img in images]
        if self.hue[0] != self.hue[1]:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            images = [TF.adjust_hue(img, hue_factor) for img in images]
        return {'img': images, 'label': sample['label']}


class ToTensor():
    def __call__(self, sample):
        images = np.stack(sample['img'])
        images = np.transpose(images, (0, 3, 1, 2)) / 255.
        images = torch.from_numpy(images).float()
        return {'img': images, 'label': sample['label']}


def list_collate(batch):
    items = {'img': [], 'label': [], 'info': []}
    for b in batch:
        items['img'].append(b['img'])
        items['label'].append(b['label'])
        items['info'].append(b['info'])
    items['img'] = default_collate(items['img'])

    return items
