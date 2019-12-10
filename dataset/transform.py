from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from torch.utils.data.dataloader import default_collate
import random
import numpy as np
from PIL import Image, ImageOps
from utils.bbox import bbox_iou_numpy


class Resize_Pad():
    def __init__(self, size, keep_ratio=True):
        '''size: w,h'''
        assert isinstance(size, (int, list, tuple))
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, sample):
        img = sample['img']
        label = sample['label']
        w, h = img.size
        net_w, net_h = self.size
        if self.keep_ratio:
            if w == net_w and h == net_h:
                # if len(label):
                #     label = label[label[:, 3] > (16. / net_w)]
                #     label = label[label[:, 4] > (16. / net_h)]
                return {'img': img, 'label': label}
            scale = min(net_h / h, net_w / w)
            if scale != 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
                w, h = img.size
            if w == net_w and h == net_h:
                # if len(label):
                #     label = label[label[:, 3] > (16. / net_w)]
                #     label = label[label[:, 4] > (16. / net_h)]
                return {'img': img, 'label': label}
            pad_w = (net_w - w) / 2
            pad_h = (net_h - h) / 2
            pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
            img = ImageOps.expand(img, border=pad, fill=(127, 127, 127))

            if not len(label):
                return {'img': img, 'label': label}

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
            # label = label[label[:, 3] > (16. / net_w)]
            # label = label[label[:, 4] > (16. / net_h)]
        else:
            img = img.resize(self.size, Image.NEAREST)

        return {'img': img, 'label': label}


class RandomCrop():
    def __init__(self, jitter):
        assert jitter < 0.5
        self.jitter = jitter

    def __call__(self, sample):
        if not len(sample['label']):
            return sample
        img = sample['img']
        label_org = sample['label']
        w, h = img.size
        dw, dh = int(w * self.jitter), int(h * self.jitter)
        while True:
            label = np.array(label_org)
            crop_left = random.randint(-dw, dw)
            crop_right = random.randint(-dw, dw)
            crop_top = random.randint(-dh, dh)
            crop_bottom = random.randint(-dh, dh)
            j = max(0, crop_left)
            i = max(0, crop_top)
            tw = w - 1 - max(0, crop_right) - j
            th = h - 1 - max(0, crop_bottom) - i

            center = label[:, 1:3]
            mask = (center[:, 0] * w > j) * (
                    center[:, 1] * h > i) * (center[:, 0] * w < j + tw) * (
                           center[:, 1] * h < i + th)
            if not mask.any():
                continue
            # label = label[mask]
            x1 = w * (label[:, 1] - label[:, 3] / 2)
            y1 = h * (label[:, 2] - label[:, 4] / 2)
            x2 = w * (label[:, 1] + label[:, 3] / 2)
            y2 = h * (label[:, 2] + label[:, 4] / 2)
            x1 = np.min((np.array([tw] * len(label)), np.max((np.zeros(len(label)), x1.T - j), axis=0)), axis=0)
            y1 = np.min((np.array([th] * len(label)), np.max((np.zeros(len(label)), y1.T - i), axis=0)), axis=0)
            x2 = np.min((np.array([tw] * len(label)), np.max((np.zeros(len(label)), x2.T - j), axis=0)), axis=0)
            y2 = np.min((np.array([th] * len(label)), np.max((np.zeros(len(label)), y2.T - i), axis=0)), axis=0)
            label[:, 1] = ((x1 + x2) / 2) / (tw + 1)
            label[:, 2] = ((y1 + y2) / 2) / (th + 1)
            label[:, 3] = (x2 - x1) / (tw + 1)
            label[:, 4] = (y2 - y1) / (th + 1)
            label = label[label[:, 3] > (2 / tw)]
            label = label[label[:, 4] > (2 / th)]

            return {'img': TF.crop(img, i, j, th, tw), 'label': label}


class HFlip():
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, sample):
        if not len(sample['label']):
            return sample
        img = sample['img']
        label = sample['label']
        w, h = img.size
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label[:, 1] = ((1 - label[:, 1]) * w - 1) / w

        return {'img': img, 'label': label}


class ColorJitter():
    def __init__(self, contrast=1., saturation=1., hue=0.):
        self.contrast = [2 - contrast, contrast]
        self.saturation = [2 - saturation, saturation]
        self.hue = [-hue, hue]

    def __call__(self, sample):
        if not len(sample['label']):
            return sample
        img = sample['img']
        if self.contrast[0] != self.contrast[1]:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            img = TF.adjust_contrast(img, contrast_factor)
        if self.saturation[0] != self.saturation[1]:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            img = TF.adjust_saturation(img, saturation_factor)
        if self.hue[0] != self.hue[1]:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            img = TF.adjust_hue(img, hue_factor)
        return {'img': img, 'label': sample['label']}


class ToTensor():
    def __call__(self, sample):
        return {'img': TF.to_tensor(sample['img']), 'label': sample['label']}


class RC():
    def __init__(self, size, jitter, fill_color=127):
        self.output_w, self.output_h = size
        self.jitter = jitter
        self.fill_color = fill_color

    def __call__(self, sample):
        """ Take random crop from image """
        img = sample['img']
        label_org = sample['label']
        orig_w, orig_h = img.size
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)
        while True:
            new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
            scale = random.random() * (2 - 0.25) + 0.25  # 0.25~2
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
            orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
            orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
            output_img = Image.new(img.mode, (self.output_w, self.output_h), color=(self.fill_color,) * channels)
            output_img.paste(orig_crop_resize, (0, 0))
            self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]
            sx, sy, crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.crop_info
            label = np.array(label_org)
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
            if len(label) == 0: continue
            label[:, 1:5:2] /= self.output_w
            label[:, 2:5:2] /= self.output_h
            return {'img': output_img, 'label': label}


class RCM():
    def __init__(self, size, jitter, fill_color=127):
        self.output_w, self.output_h = size
        self.jitter = jitter
        self.fill_color = fill_color

    def __call__(self, sample):
        """ Take random crop from image """
        img = sample['img']
        label_org = sample['label']
        # print('output shape: %d, %d' % (self.output_w, self.output_h))
        orig_w, orig_h = img.size
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)
        while True:
            new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
            scale = random.random() * (2 - 0.25) + 0.25  # 0.25~2
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
            orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
            label = np.array(label_org)
            label[:, 1:5:2] *= orig_w
            label[:, 2:5:2] *= orig_h
            x1 = np.maximum(orig_xmin, (label[:, 1] - label[:, 3] / 2))
            x2 = np.minimum(orig_xmax, (label[:, 1] + label[:, 3] / 2))
            y1 = np.maximum(orig_ymin, (label[:, 2] - label[:, 4] / 2))
            y2 = np.minimum(orig_ymax, (label[:, 2] + label[:, 4] / 2))
            label[:, 3] = x2 - x1
            label[:, 4] = y2 - y1
            label[:, 1] = x1 - orig_xmin + label[:, 3] / 2
            label[:, 2] = y1 - orig_ymin + label[:, 4] / 2
            label = label[label[:, 3] > 2]
            label = label[label[:, 4] > 2]
            if len(label) == 0: continue
            label[:, 1:5:2] /= (orig_xmax - orig_xmin)
            label[:, 2:5:2] /= (orig_ymax - orig_ymin)
            return {'img': orig_crop, 'label': label}


class SSDCrop(object):

    def __init__(self, min_ious=(0.3, 0.5, 0.7, 0.9), min_crop_size=0.4):
        # 1: return ori img
        # self.large_mode = (1, 1, 1, 0.5, 0.7, 0.9)
        # self.small_mode = (1, 0.1, 0.1, 0.3, 0.3, 0)
        self.mode = (0, *min_ious, 1)
        self.min_crop_size = min_crop_size

    def __call__(self, sample):
        img = sample['img']
        labels = sample['label']
        w, h = img.size
        while True:
            # sample_mode = self.large_mode
            sample_mode = self.mode

            labels_cp = np.array(labels)
            labels_cp[:, 1::2] *= w
            labels_cp[:, 2::2] *= h

            boxes = np.array(labels_cp[:, 1:])
            boxes[:, :2] = labels_cp[:, 1:3] - labels_cp[:, -2:] / 2
            boxes[:, -2:] = labels_cp[:, 1:3] + labels_cp[:, -2:] / 2
            # if np.min(boxes[:, -2:]) < 0.3 * min(w, h): sample_mode = self.small_mode
            mode = random.choice(sample_mode)
            if mode == 1:
                return {'img': img, 'label': labels}

            min_iou = mode
            for i in range(50):
                if min_iou and i > 10: continue
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w - 1), int(top + new_h - 1)))
                overlaps = bbox_iou_numpy(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue
                if not min_iou and overlaps.max() < 0.7:
                    continue

                # center of boxes should inside the crop img
                center = labels_cp[:, 1:3]
                mask = (center[:, 0] > patch[0]) * (
                        center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                               center[:, 1] < patch[3])
                if not mask.any():
                    continue
                labels_cp = labels_cp[mask]

                # adjust boxes
                img = img.crop(patch)
                x1 = np.maximum(patch[0], (labels_cp[:, 1] - labels_cp[:, 3] / 2))
                x2 = np.minimum(patch[2], (labels_cp[:, 1] + labels_cp[:, 3] / 2))
                y1 = np.maximum(patch[1], (labels_cp[:, 2] - labels_cp[:, 4] / 2))
                y2 = np.minimum(patch[3], (labels_cp[:, 2] + labels_cp[:, 4] / 2))
                labels_cp[:, 3] = x2 - x1
                labels_cp[:, 4] = y2 - y1
                labels_cp[:, 1] = x1 - patch[0] + labels_cp[:, 3] / 2
                labels_cp[:, 2] = y1 - patch[1] + labels_cp[:, 4] / 2

                labels_cp[:, 1::2] /= new_w
                labels_cp[:, 2::2] /= new_h

                return {'img': img, 'label': labels_cp}


class Normalize():
    def __init__(self, mean, std, to_bgr255=True):
        self.to_bgr255 = to_bgr255
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['img'] * 255.
        if self.to_bgr255:
            img = img[[2, 1, 0]]
        img = TF.normalize(img, self.mean, self.std)
        return {'img': img, 'label': sample['label']}


def list_collate(batch):
    items = {'img': [], 'label': [], 'info': []}
    for b in batch:
        items['img'].append(b['img'])
        items['label'].append(b['label'])
        items['info'].append(b['info'])
    items['img'] = default_collate(items['img'])

    return items
