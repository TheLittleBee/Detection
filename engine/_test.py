from littlenet import models
import dataset as dt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class TestEngine():
    def __init__(self, hyper_params):
        self.batch_size = hyper_params.batch

        self.n_cls = hyper_params.n_cls
        self.classes = hyper_params.classes
        root = hyper_params.root

        self.cuda = True if torch.cuda.is_available() else False

        self.model_name = hyper_params.model_name
        model_cfg = hyper_params.model_cfg
        model_type = model_cfg.pop('type')
        assert model_type in models.__dict__
        self.bg = model_cfg.pop('background')
        model_cfg['weights'] = hyper_params.weights
        model_cfg['head']['num_classes'] = self.n_cls + 1 if self.bg else self.n_cls
        self.net = models.__dict__[model_type](**model_cfg)
        setattr(self.net, 'num_classes', model_cfg['head']['num_classes'])

        if self.cuda:
            self.net.cuda()
        self.net.eval()

        self.input_size = hyper_params.input_size
        setattr(self.net, 'input_size', self.input_size)
        transform = [dt.Resize_Pad(self.input_size), dt.ToTensor()]
        if hyper_params.norm:
            transform += [dt.Normalize(hyper_params.norm['mean'], hyper_params.norm['std'])]
        ann_file = root + hyper_params.ann_file
        img_dir = root + hyper_params.img_dir
        if hasattr(hyper_params, 'json'):
            self.dataset = dt.JsonDataset(ann_file, img_dir, transform=transforms.Compose(transform),
                                          city=hyper_params.city)
        else:
            self.dataset = dt.VOCDataset(ann_file, img_dir, transform=transforms.Compose(transform))
        setattr(self.dataset, 'with_bg', self.bg)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=hyper_params.nworkers if self.cuda else 0,
            # pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=dt.list_collate,
        )

        self.conf_thresh = hyper_params.conf_th
        self.nms_thresh = hyper_params.nms_th
        self.ignore_thresh = hyper_params.ignore_th
        self.n_det = hyper_params.n_det
