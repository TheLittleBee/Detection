from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import os
import pickle

from .voc_eval import voceval
from utils.bbox import label_to_box

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VIDDataset(Dataset):
    with_bg = False

    def __init__(self, ann_file, img_dir, seq=10, skip=1, transform=None):
        if isinstance(ann_file, (list, tuple)):
            self.annos = {}
            for file in ann_file:
                with open(file, 'rb') as f:
                    self.annos.update(pickle.load(f))
        else:
            with open(ann_file, 'rb') as f:
                self.annos = pickle.load(f)
        self.keys = list(self.annos)
        self.frames = [len(self.annos[k]) // (seq * skip) * skip for k in self.keys]
        self.img_dir = img_dir
        self.seq = seq
        self.skip = skip
        self.transform = transform
        self.img_size = {}

    def __getitem__(self, index):
        v = 0
        while index >= self.frames[v]:
            index -= self.frames[v]
            v += 1
        images = []
        vid = self.keys[v]
        id = index // self.skip * self.seq * self.skip + index % self.skip
        for i in range(id, id + self.seq * self.skip, self.skip):
            img_path = os.path.join(self.img_dir, vid, '{:0>6}.JPEG'.format(i))
            img = Image.open(img_path)
            images.append(img)
        self.img_size[vid] = images[0].size
        labels = [np.array(self.annos[vid][i]) for i in range(id, id + self.seq * self.skip, self.skip)]
        if self.with_bg:
            for l in labels:
                l[:, 0] += 1

        sample = {'img': images, 'label': labels}
        if self.transform:
            sample = self.transform(sample)
        sample['info'] = {'id': vid}
        return sample

    def __len__(self):
        return np.sum(self.frames)

    def eval(self, n_cls, pred):
        dets = []
        annos = []
        for k in pred:
            dets += pred[k]
            annos += [label_to_box(np.array(ann), self.img_size[k]) for ann in self.annos[k]]
        ap, p, r = voceval(n_cls, dets, annos)
        res = {'mAP': np.mean(ap)}
        for i, v in enumerate(ap):
            res[i] = (v, p[i], r[i])
        mp = np.mean(p)
        res['mP'] = mp
        mr = np.mean(r)
        res['mR'] = mr
        res['F1'] = 2 * mp * mr / (mp + mr)
        return res
