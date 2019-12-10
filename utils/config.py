import yaml
import torch
import numpy as np
import random
from os import path

JSONDATA = ['COCO', 'CITY']


def yml_parse(file):
    with open(file, 'r') as fp:
        y = yaml.load(fp)
        return y


class HyperParams():
    def __init__(self, args, train_flag):
        self.model_name = path.basename(args.model).split('.')[0]
        self.data_name = args.data

        model_cfg = yml_parse(args.model)
        data_name = yml_parse('cfg/dataset.yml')
        if self.data_name not in data_name.keys():
            raise Exception
        data_cfg = data_name[self.data_name]
        if self.data_name in JSONDATA:
            self.json = True
            self.city = False
            if self.data_name == JSONDATA[1]:
                self.city = True
        self.n_cls = data_cfg['classes_num']
        self.classes = data_cfg['labels']
        self.root = data_cfg['root']
        self.norm = model_cfg.get('norm')

        self.model_cfg = model_cfg['model']

        if train_flag == 1:
            model_cfg = model_cfg['train']
            self.nworkers = model_cfg['nworkers']
            self.pin_mem = model_cfg['pin_mem']

            self.input_size = model_cfg['input_size']

            self.batch = model_cfg['batch_size']
            self.mini_batch = model_cfg['mini_batch_size']
            self.max_batches = model_cfg['max_batches']

            data_train = data_cfg['train']
            self.img_dir = data_train['image_directory']
            self.ann_file = data_train['annotation_file']
            for k, v in data_train['transform'].items():
                setattr(self, k, v)

            self.optim_cfg = model_cfg['optimizer']
            self.lr_cfg = model_cfg['scheduler']

            self.backup = model_cfg['backup_interval']
            self.bp_steps = model_cfg['backup_steps']
            self.bp_rates = model_cfg['backup_rates']
            self.backup_dir = model_cfg['backup_dir']

            self.resize = model_cfg['resize_interval']
            self.rs_steps = []
            self.rs_rates = []

            if args.weight is not None and args.weight[-4:] == 'ckpt':
                self.weights = model_cfg['weights']
                self.ckpt = args.weight
            else:
                self.weights = args.weight if args.weight else model_cfg['weights']
                self.ckpt = None
            self.model_cfg['clear'] = model_cfg['clear']

            if args.val:
                data_val = data_cfg['test']
                self.val = {'img_dir': data_val['image_directory'],
                            'ann_file': data_val['annotation_file']}

        elif train_flag == 2:
            model_cfg = model_cfg['test']
            self.nworkers = model_cfg['nworkers']
            self.pin_mem = model_cfg['pin_mem']

            self.input_size = model_cfg['input_size']

            data_cfg = data_cfg['test']
            self.img_dir = data_cfg['image_directory']
            self.ann_file = data_cfg['annotation_file']

            self.batch = model_cfg['batch_size']

            self.conf_th = model_cfg['conf_thresh']
            self.nms_th = model_cfg['nms_thresh']
            self.ignore_th = model_cfg['ignore_thresh']
            self.n_det = model_cfg['num_detect']

            self.weights = args.weight if args.weight else model_cfg['weights']


def randomSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
