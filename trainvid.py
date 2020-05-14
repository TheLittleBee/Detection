import torch
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import argparse
import time
import gc
from tqdm import tqdm
import os.path as path
import yaml

from utils.config import randomSeed
from utils.logger import setup_logger
import VID.dataset as dt
from VID.models.yolo import YOLO
from engine.lr_scheduler import *

trainanns = '/home/littlebee/dataset/VID/ILSVRC2015/annotations_train.pkl'
trainimg = '/home/littlebee/dataset/VID/ILSVRC2015/Data/VID/train'
valanns = '/home/littlebee/dataset/VID/ILSVRC2015/annotations_val.pkl'
valimg = '/home/littlebee/dataset/VID/ILSVRC2015/Data/VID/val'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--weight', dest='weight', type=str, help='Initial weight file')
    parser.add_argument('--seq', type=int, default=10, help='Sequence number')
    parser.add_argument('--skip', type=int, default=2, help='Skip frames')
    parser.add_argument('-v', '--visual', dest='visual', action='store_true', help='Use Visdom to log')
    parser.add_argument('-b', '--board', dest='board', action='store_true', help='Use tensorboard to log')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=0, help='Set random seed')
    parser.add_argument('--val', dest='val', action='store_true', help='Val when training')
    args = parser.parse_args()
    return args


args = parse_args()
cuda = torch.cuda.is_available()
model_name = path.basename(args.model).split('.')[0]
backup_dir = 'backup/'
nworkers = 8
logger = setup_logger(model_name, 'log/')
logger.info(args)

if args.seed: randomSeed(args.seed)
visual = None
if args.visual:
    from utils.logger import VVisual

    visual = VVisual(model_name + '_' + 'VID')
board = None
if args.board:
    from tensorboardX import SummaryWriter

    board = SummaryWriter(logdir='log/%s/%s/' % ('VID', model_name))

with open(args.model, 'r') as fp:
    cfg = yaml.load(fp)
model_cfg = cfg['model']
train_cfg = cfg['train']
assert model_cfg.pop('type') == 'YOLO'
bg = model_cfg.pop('background')
model_cfg['weights'] = train_cfg['weights']
model_cfg['head']['num_classes'] = 31 if bg else 30
net = YOLO(**model_cfg)
if cuda:
    net.cuda()
net.train()

lr = train_cfg['lr']
momentum = train_cfg['momentum']
weight_decay = train_cfg['weight_decay']
warmup_iters = train_cfg['warmup_iters']
milestones = train_cfg['milestones']
max_batches = train_cfg['max_batches']
batch_size = train_cfg['batch_size']

if args.seq == 1:
    transform = tf.Compose([dt.HFlip(), dt.ColorJitter(1.5, 1.5, .1), dt.RC((416, 416), 0.3), dt.ToTensor()])
else:
    transform = tf.Compose([dt.HFlip(), dt.ColorJitter(1.5, 1.5, .1), dt.Resize_Pad(416), dt.ToTensor()])
traindata = dt.VIDDataset(trainanns, trainimg, args.seq, args.skip, transform)
dataloader = DataLoader(
    traindata,
    batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=nworkers if cuda else 0,
    pin_memory=True if cuda else False,
    collate_fn=dt.list_collate,
)
if args.val:
    transform = tf.Compose([dt.Resize_Pad(416), dt.ToTensor()])
    valdata = dt.VIDDataset(valanns, valimg, args.seq, args.skip, transform)
    valloader = DataLoader(
        valdata,
        batch_size=batch_size,
        num_workers=nworkers if cuda else 0,
        pin_memory=True if cuda else False,
        collate_fn=dt.list_collate,
    )
ndata = len(dataloader)

optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = WarmupMultiStepLR(optim, milestones, warmup_epoch=warmup_iters)
if args.weight is not None:
    if args.weight.endswith('.ckpt'):
        state = torch.load(args.weight)
        net.seen = state['seen']
        net.load_state_dict(state['net'])
        optim.load_state_dict(state['optim'])
        scheduler.load_state_dict(state['sche'])
    elif args.weight.endswith('.pth'):
        state = torch.load(args.weight, map_location='cpu')
        net.load_state_dict(state['weights'], strict=False)
sepoch = round(net.seen / batch_size / ndata)
epochs = round(max_batches / ndata)

logger.info('Start training!')
for epoch in range(sepoch, epochs):
    epochtic = time.time()
    optim.zero_grad()
    losslog = {}
    with tqdm(dataloader) as loader:
        for batch_i, sample in enumerate(loader):
            nbatch = batch_i + ndata * epoch
            scheduler.step()
            imgs, targets = sample['img'], sample['label']
            if cuda:
                imgs = imgs.cuda()
            net.clear()
            loss = net(imgs, targets)

            loss.backward()

            if len(losslog) == 0:
                losslog.update(net.log)
            else:
                for k in losslog.keys():
                    losslog[k] = (losslog[k] * batch_i + net.log[k]) / (batch_i + 1)
            logstr = 'lr: %f' % optim.param_groups[0]['lr']
            for k, v in net.log.items():
                logstr += ' %s: %.4f' % (k, v)
            loader.set_description('Epoch %d' % epoch, False)
            loader.set_postfix_str(logstr, False)

            optim.step()
            optim.zero_grad()

            if nbatch % 100 == 0 or batch_i + 1 == ndata:
                net.save_weights('{}{}_{}_fine.pth'.format(backup_dir, model_name, 'VID'))
                if visual:
                    visual.line(losslog, nbatch)

                if board:
                    for k, v in losslog.items():
                        board.add_scalar(k, v, nbatch)

    epochtoc = time.time()

    state = {
        'seen': net.seen,
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'sche': scheduler.state_dict(),
    }
    torch.save(state, '{}{}_{}.ckpt'.format(backup_dir, model_name, 'VID'))
    logstr = 'lr: %f' % optim.param_groups[0]['lr']
    for k, v in losslog.items():
        logstr += ' %s: %.4f' % (k, v)
    logger.info('Epoch %d finished, ' % (epoch) + logstr)

    net.save_weights('{}{}_{}_{}.pth'.format(backup_dir, model_name, 'VID', epoch))

    if args.val and epoch % 2 == 0:
        valloss = 0
        for sample in valloader:
            img, target = sample['img'], sample['label']
            if cuda: img = img.cuda()
            net.clear()
            with torch.no_grad():
                valloss += net(img, target).item()
        logger.info('Epoch %d val loss: %.4f' % (epoch, valloss / len(valloader)))
        if visual:
            visual.line({'val': valloss / len(valloader)}, nbatch)
        if board:
            board.add_scalar('val', valloss / len(valloader), nbatch)
    gc.collect()
board.close()
net.save_weights('{}{}_{}_final.pth'.format(backup_dir, model_name, 'VID'))
