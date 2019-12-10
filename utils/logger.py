# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import numpy as np
try:
    from visdom import Visdom
    print('Have visdom installed.')
except:
    print('NO visdom')


def setup_logger(name, save_dir, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class VVisual():
    def __init__(self, env):
        self.viz = Visdom(env=env)
        assert self.viz.check_connection(), 'No Visdom Connection'

    def line(self, log, x):
        try:
            for k, v in log.items():
                self.viz.line(Y=np.array([v]),
                              X=np.array([x]),
                              win=k,
                              update='append' if x else None,
                              opts={'title': k})
        except BaseException as e:
            print('Visdom exception: {}'.format(repr(e)))

