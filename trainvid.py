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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--weight', dest='weight', type=str, help='Initial weight file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seq', type=int, default=10, help='Sequence number')
    parser.add_argument('--skip', type=int, default=2, help='Skip frames')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
backup = 10
nworkers = 4
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
assert model_cfg.pop('type') == 'YOLO'
bg = model_cfg.pop('background')
model_cfg['weights'] = None if args.weight.endswith('.ckpt') else args.weight
model_cfg['head']['num_classes'] = 31 if bg else 30
net = YOLO(**model_cfg)
if cuda: net.cuda()
net.train()

trainanns = '/home/littlebee/dataset/VID/ILSVRC2015/annotations_train.pkl'
trainimg = '/home/littlebee/dataset/VID/ILSVRC2015/Data/VID/train'
transform = tf.Compose([dt.HFlip(), dt.Resize_Pad(416), dt.ToTensor()])
traindata = dt.VIDDataset(trainanns, trainimg, args.seq, args.skip, transform)
dataloader = DataLoader(
    traindata,
    args.batch,
    shuffle=True,
    drop_last=True,
    num_workers=nworkers if cuda else 0,
    # pin_memory=pin_mem if cuda else False,
    collate_fn=dt.list_collate,
)
if args.val:
    valanns = '/home/littlebee/dataset/VID/ILSVRC2015/annotations_val.pkl'
    valimg = '/home/littlebee/dataset/VID/ILSVRC2015/Data/VID/val'
    transform = tf.Compose([dt.Resize_Pad(416), dt.ToTensor()])
    valdata = dt.VIDDataset(valanns, valimg, args.seq, args.skip, transform)
    valloader = DataLoader(
        valdata,
        batch_size=args.batch,
        num_workers=nworkers if cuda else 0,
        # pin_memory=pin_mem if cuda else False,
        collate_fn=dt.list_collate,
    )

optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
scheduler = WarmupMultiStepLR(optim, [round(args.epochs * 0.8)], warmup_epoch=5)
if args.weight.endswith('.ckpt'):
    state = torch.load(args.weight)
    net.seen = state['seen']
    net.load_state_dict(state['net'])
    optim.load_state_dict(state['optim'])
    scheduler.load_state_dict(state['sche'])
sepoch = round(net.seen / args.batch / len(dataloader))

logger.info('Start training!')
for epoch in range(sepoch, args.epochs):
    epochtic = time.time()
    scheduler.step()
    optim.zero_grad()
    losslog = {}
    with tqdm(dataloader) as loader:
        for batch_i, sample in enumerate(loader):
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
                    losslog[k] += net.log[k]
            logstr = 'lr: %f' % optim.param_groups[0]['lr']
            for k, v in net.log.items():
                logstr += ' %s: %.4f' % (k, v)
            loader.set_description('Epoch %d' % epoch, False)
            loader.set_postfix_str(logstr, False)

            optim.step()
            optim.zero_grad()

            if batch_i % 100 == 0:
                net.save_weights('{}{}_{}_fine.pth'.format(backup_dir, model_name, 'VID'))

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
        logstr += ' %s: %.4f' % (k, v / len(dataloader))
    logger.info('Epoch %d finished, ' % (epoch) + logstr)
    if visual:
        visual.line(losslog, epoch)

    if board:
        for k, v in losslog.items():
            board.add_scalar(k, v, epoch)

    if epoch % backup == 0:
        net.save_weights('{}{}_{}_{}.pth'.format(backup_dir, model_name, 'VID', epoch))

    if args.val and epoch % 20 == 0:
        valloss = 0
        for sample in valloader:
            img, target = sample['img'], sample['label']
            if cuda: img = img.cuda()
            with torch.no_grad():
                valloss += net(img, target).item()
        logger.info('Epoch %d val loss: %.4f' % (epoch, valloss / len(valloader)))
        if visual:
            visual.line({'val': valloss / len(valloader)}, epoch)
        if board:
            board.add_scalar('val', valloss / len(valloader), epoch)
    gc.collect()
board.close()
net.save_weights('{}{}_{}_final.pth'.format(backup_dir, model_name, 'VID'))
