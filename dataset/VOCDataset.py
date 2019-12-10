from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import os.path
import pickle
from .voc_eval import voceval

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VOCDataset(Dataset):
    with_bg = False

    def __init__(self, ann_file, img_dir, transform=None):
        if isinstance(ann_file, (list, tuple)):
            self.annos = {}
            for file in ann_file:
                with open(file, 'rb') as f:
                    self.annos.update(pickle.load(f))
        else:
            with open(ann_file, 'rb') as f:
                self.annos = pickle.load(f)
        self.keys = list(self.annos)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = {}

    def __getitem__(self, index):
        img_path = self.img_dir + self.keys[index]

        img = Image.open(img_path)
        self.img_size[self.keys[index]] = img.size
        labels = np.array(self.annos[self.keys[index]])
        if self.with_bg:
            labels[:, 0] += 1

        sample = {'img': img, 'label': labels}
        if self.transform:
            sample = self.transform(sample)
        sample['info'] = {'path': img_path, 'id': self.keys[index]}
        return sample

    def __len__(self):
        return len(self.keys)

    def eval(self, n_cls, dets, ids, input_size, resFile):
        ap, p, r = voceval(n_cls, dets, ids, self, input_size)
        res = {'mAP': np.mean(ap)}
        for i, v in enumerate(ap):
            res[i] = (v, p[i], r[i])
        mp = np.mean(p)
        res['mP'] = mp
        mr = np.mean(r)
        res['mR'] = mr
        res['F1'] = 2 * mp * mr / (mp + mr)
        return res
