import torchvision.transforms.functional as TF
import torch
from torch.utils.data.dataloader import default_collate
import random
import numpy as np
from PIL import Image, ImageOps

__all__ = ['Resize_Pad', 'HFlip', 'ColorJitter', 'ToTensor', 'RC', 'list_collate']


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


class RC():
    def __init__(self, size, jitter, fill_color=127):
        self.output_w, self.output_h = size
        self.jitter = jitter
        self.fill_color = fill_color

    def __call__(self, sample):
        """ Take random crop from image """
        if isinstance(sample['img'], list):
            images = sample['img']
            labels = sample['label']
        else:
            images = [sample['img']]
            labels = [sample['label']]
        orig_w, orig_h = images[0].size
        channels = 3
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)
        new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
        scale = random.random() * (1.5 - 0.5) + 0.5  # 0.5~1.5
        if new_ar < 1:
            nh = int(scale * orig_h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * orig_w)
            nh = int(nw / new_ar)

        if self.output_w > nw:
            dx = random.randint(0, self.output_w - nw)
        else:
            dx = random.randint(self.output_w - nw, 0)

        if self.output_h > nh:
            dy = random.randint(0, self.output_h - nh)
        else:
            dy = random.randint(self.output_h - nh, 0)

        nxmin = max(0, -dx)
        nymin = max(0, -dy)
        nxmax = min(nw, -dx + self.output_w - 1)
        nymax = min(nh, -dy + self.output_h - 1)
        sx, sy = float(orig_w) / nw, float(orig_h) / nh
        orig_xmin = int(nxmin * sx)
        orig_ymin = int(nymin * sy)
        orig_xmax = int(nxmax * sx)
        orig_ymax = int(nymax * sy)
        self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]
        output_imgs = []
        output_labels = []
        for img, label in zip(images, labels):
            orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
            orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
            output_img = Image.new(img.mode, (self.output_w, self.output_h), color=(self.fill_color,) * channels)
            output_img.paste(orig_crop_resize, (0, 0))
            output_imgs.append(output_img)
            sx, sy, crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.crop_info
            if len(label) == 0:
                output_labels.append(label)
                continue
            label[:, 1:5:2] *= orig_w
            label[:, 2:5:2] *= orig_h
            # label[:,1] -= orig_xmin
            # label[:,2] -= orig_ymin
            # label[:,1:5:2] *= (nxmax-nxmin)/((orig_xmax-orig_xmin)*self.output_w)
            # label[:,2:5:2] *= (nymax-nymin)/((orig_ymax-orig_ymin)*self.output_h)
            x1 = np.maximum(crop_xmin, (label[:, 1] - label[:, 3] / 2) / sx)
            x2 = np.minimum(crop_xmax, (label[:, 1] + label[:, 3] / 2) / sx)
            y1 = np.maximum(crop_ymin, (label[:, 2] - label[:, 4] / 2) / sy)
            y2 = np.minimum(crop_ymax, (label[:, 2] + label[:, 4] / 2) / sy)
            label[:, 3] = x2 - x1
            label[:, 4] = y2 - y1
            label[:, 1] = x1 - crop_xmin + label[:, 3] / 2
            label[:, 2] = y1 - crop_ymin + label[:, 4] / 2
            label = label[label[:, 3] > 2]
            label = label[label[:, 4] > 2]
            label[:, 1:5:2] /= self.output_w
            label[:, 2:5:2] /= self.output_h
            output_labels.append(label)
        return {'img': output_imgs, 'label': output_labels}


def list_collate(batch):
    items = {'img': [], 'label': [], 'info': []}
    for b in batch:
        items['img'].append(b['img'])
        items['label'].append(b['label'])
        items['info'].append(b['info'])
    items['img'] = default_collate(items['img'])

    return items
