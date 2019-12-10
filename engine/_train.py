from littlenet import models
import dataset as dt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import signal
from .lr_scheduler import *


class TrainEngine():
    def __init__(self, hyper_params):
        # get config
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        # subdivision using accumulated grad
        self.subdivision = self.batch_size // self.mini_batch_size

        self.n_cls = hyper_params.n_cls
        self.classes = hyper_params.classes
        self.root = hyper_params.root

        self.cuda = True if torch.cuda.is_available() else False
        self.backup_dir = hyper_params.backup_dir
        self.backup = hyper_params.backup

        # listening SIGINT while Ctrl + c
        self.sigint = False
        signal.signal(signal.SIGINT, self.__sigint_handler)

        self.model_name = hyper_params.model_name
        model_cfg = hyper_params.model_cfg
        model_type = model_cfg.pop('type')
        assert model_type in models.__dict__
        self.bg = model_cfg.pop('background')
        model_cfg['weights'] = hyper_params.weights
        model_cfg['head']['num_classes'] = self.n_cls + 1 if self.bg else self.n_cls
        self.net = models.__dict__[model_type](**model_cfg)

        if self.cuda:
            self.net.cuda()
        self.net.train()

        self.input_size = hyper_params.input_size
        setattr(self.net, 'input_size', self.input_size)
        transform = []
        if hasattr(hyper_params, 'crop'):
            # transform.append(dt.RCM(self.input_size,hyper_params.crop))
            transform.append(dt.RC(self.input_size,hyper_params.crop))
            # transform.append(dt.RandomCrop(hyper_params.crop))
            # transform.append(dt.SSDCrop())
        if hasattr(hyper_params, 'flip'):
            transform.append(dt.HFlip(hyper_params.flip))
        if hasattr(hyper_params, 'hue'):
            transform.append(dt.ColorJitter(hyper_params.exposure, hyper_params.saturation, hyper_params.hue))
        transform += [dt.Resize_Pad(self.input_size), dt.ToTensor()]
        if hyper_params.norm:
            transform += [dt.Normalize(hyper_params.norm['mean'], hyper_params.norm['std'])]
        ann_file = self.root + hyper_params.ann_file
        img_dir = self.root + hyper_params.img_dir
        if hasattr(hyper_params, 'json'):
            dataset = dt.JsonDataset(ann_file, img_dir, transform=transforms.Compose(transform), city=hyper_params.city)
        else:
            dataset = dt.VOCDataset(ann_file, img_dir, transform=transforms.Compose(transform))
        setattr(dataset, 'with_bg', self.bg)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=hyper_params.nworkers if self.cuda else 0,
            # pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=dt.list_collate,
        )
        if hasattr(hyper_params, 'val'):
            ann_file = self.root + hyper_params.val['ann_file']
            img_dir = self.root + hyper_params.val['img_dir']
            transform = [dt.Resize_Pad(self.input_size), dt.ToTensor()]
            if hyper_params.norm:
                transform += [dt.Normalize(hyper_params.norm['mean'], hyper_params.norm['std'])]
            if hasattr(hyper_params, 'json'):
                valdata = dt.JsonDataset(ann_file, img_dir, transform=transforms.Compose(transform),
                                         city=hyper_params.city)
            else:
                valdata = dt.VOCDataset(ann_file, img_dir, transform=transforms.Compose(transform))
            setattr(valdata, 'with_bg', self.bg)
            self.valloader = DataLoader(
                valdata,
                batch_size=self.mini_batch_size,
                num_workers=hyper_params.nworkers if self.cuda else 0,
                # pin_memory=hyper_params.pin_mem if self.cuda else False,
                collate_fn=dt.list_collate,
            )

        self.batch_e = len(self.dataloader) // self.subdivision
        self.max_batches = hyper_params.max_batches
        self.epoches = self.max_batches // self.batch_e

        self.optim_cfg = hyper_params.optim_cfg
        self.optim = self.make_optimizer(self.net)
        self.lr_cfg = hyper_params.lr_cfg
        self.scheduler = self.make_lr_scheduler(self.optim)

        if hyper_params.ckpt is not None:
            state = torch.load(hyper_params.ckpt)
            self.net.seen = state['seen']
            self.net.load_state_dict(state['net'])
            self.optim.load_state_dict(state['optim'])
            self.scheduler.load_state_dict(state['sche'])

        self.log = {}

    @property
    def batch(self):
        return self.net.seen // self.batch_size

    @property
    def lr(self):
        return self.optim.param_groups[0]['lr']

    def dolog(self):
        if len(self.log) == 0:
            self.log = self.net.log
        else:
            for k in self.log.keys():
                self.log[k] += self.net.log[k]

    def getlog(self):
        l = self.log
        for k in l.keys():
            l[k] /= self.batch_e * self.subdivision
        l['learning_rate'] = self.lr
        self.log = {}
        return l

    def checkpoint(self, path):
        state = {
            'seen': self.net.seen,
            'net': self.net.state_dict(),
            'optim': self.optim.state_dict(),
            'sche': self.scheduler.state_dict(),
        }
        torch.save(state, path)

    def __sigint_handler(self, signal, frame):
        if not self.sigint:
            print('SIGINT caught. Waiting for gracefull exit')
            self.sigint = True

    def quit(self, model, data):
        if self.sigint:
            self.net.save_weights(self.backup_dir + model + '_' + data + '_fine.pth')
            return True
        elif self.batch >= self.max_batches:
            self.net.save_weights(self.backup_dir + model + '_' + data + '_final.pth')
            return True
        else:
            return False

    def make_optimizer(self, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.optim_cfg['base_lr']
            weight_decay = self.optim_cfg['weight_decay']
            if "bias" in key:
                lr = self.optim_cfg['base_lr'] * self.optim_cfg['bias_lr_factor']
                weight_decay = self.optim_cfg['bias_weight_decay']
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        optimizer = torch.optim.SGD(params, lr, momentum=self.optim_cfg['momentum'])
        # optimizer = torch.optim.Adam(params, lr)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_cfg['warmup_iters']:
            warmup_epoch = self.lr_cfg.pop('warmup_iters') // self.batch_e
            if self.lr_cfg.pop('type') == 'cos':
                return WarmupCosineAnnealingLR(optimizer, self.epoches - warmup_epoch, warmup_epoch=warmup_epoch,
                                               **self.lr_cfg)
            milestones = [m // self.batch_e for m in self.lr_cfg.pop('milestones')]
            return WarmupMultiStepLR(optimizer, milestones=milestones, warmup_epoch=warmup_epoch, **self.lr_cfg)
        if self.lr_cfg['type'] == 'cos':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoches, self.lr_cfg['eta_min'])
        milestones = [m // self.batch_e for m in self.lr_cfg['milestones']]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, self.lr_cfg['gamma'])
