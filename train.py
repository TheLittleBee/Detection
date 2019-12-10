import torch
from torch.autograd import Variable
import numpy as np
import argparse
import time
import gc
from tqdm import tqdm

from utils.config import HyperParams, randomSeed
from engine._train import TrainEngine
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-d', '--data', dest='data', type=str, default='CITY', help='Dataset to use')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--weight', dest='weight', type=str, help='Initial weight file')
    parser.add_argument('-v', '--visual', dest='visual', action='store_true', help='Use Visdom to log')
    parser.add_argument('-b', '--board', dest='board', action='store_true', help='Use tensorboard to log')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=0, help='Set random seed')
    parser.add_argument('--val', dest='val', action='store_true', help='Val when training')
    args = parser.parse_args()
    return args


args = parse_args()

hyper = HyperParams(args, 1)
engine = TrainEngine(hyper)
del hyper

logger = setup_logger(engine.model_name, 'log/')
logger.info(args)

if args.seed: randomSeed(args.seed)
visual = None
if args.visual:
    from utils.logger import VVisual

    visual = VVisual(engine.model_name + '_' + args.data)
board = None
if args.board:
    from tensorboardX import SummaryWriter

    board = SummaryWriter(logdir='log/%s/%s/' % (args.data, engine.model_name))

sepoch = engine.batch // engine.batch_e

logger.info('Start training!')
for epoch in range(sepoch, engine.epoches):
    epochtic = time.time()
    engine.scheduler.step()
    engine.optim.zero_grad()
    with tqdm(engine.dataloader) as loader:
        for batch_i, sample in enumerate(loader):
            imgs, targets = sample['img'], sample['label']
            if engine.cuda:
                imgs = imgs.cuda()

            loss = engine.net(imgs, targets)

            loss.backward()
            engine.dolog()
            logstr = 'lr: %f' % engine.lr
            for k, v in engine.net.log.items():
                logstr += ' %s: %.4f' % (k, v)
            loader.set_description('Epoch %d' % epoch, False)
            loader.set_postfix_str(logstr, False)
            if (batch_i + 1) % engine.subdivision != 0:
                continue
            engine.optim.step()
            engine.optim.zero_grad()

            if engine.quit(engine.model_name, args.data):
                exit()

            if engine.batch % 100 == 0:
                engine.net.save_weights('{}{}_{}_fine.pth'.format(engine.backup_dir, engine.model_name, args.data))

            if (len(loader) - batch_i) <= engine.subdivision:
                break

    epochtoc = time.time()

    engine.checkpoint('{}{}_{}.ckpt'.format(engine.backup_dir, engine.model_name, args.data))
    log = engine.getlog()
    logstr = 'lr: %f' % engine.lr
    for k, v in log.items():
        logstr += ' %s: %.4f' % (k, v)
    logger.info('Epoch %d finished, ' % (epoch) + logstr)
    if visual:
        visual.line(log, epoch)

    if board:
        for k, v in log.items():
            board.add_scalar(k, v, epoch)

    if epoch % engine.backup == 0:
        engine.net.save_weights('{}{}_{}_{}.pth'.format(engine.backup_dir, engine.model_name, args.data, epoch))
        engine.checkpoint('{}{}_{}_bak.ckpt'.format(engine.backup_dir, engine.model_name, args.data))

    if args.val and epoch % 20 == 0:
        valloss = 0
        for sample in engine.valloader:
            img, target = sample['img'], sample['label']
            if engine.cuda: img = img.cuda()
            with torch.no_grad():
                valloss += engine.net(img, target).item()
        logger.info('Epoch %d val loss: %.4f' % (epoch, valloss / len(engine.valloader)))
        if visual:
            visual.line({'val': valloss / len(engine.valloader)}, epoch)
        if board:
            board.add_scalar('val', valloss / len(engine.valloader), epoch)
    gc.collect()
board.close()
engine.net.save_weights('{}{}_{}_final.pth'.format(engine.backup_dir, engine.model_name, args.data))
