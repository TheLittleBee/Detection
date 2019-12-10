Contents

[Requirement](#requirement) <br>
[Architecture](#architecture) <br>
[Features](#features) <br>
[Training](#training) <br>
[Evaluation](#evaluation)

---

## Requirement
- python 3.6
- pytorch >= 0.4.0
- torchvision
- pyyaml
- pycocotools
- cython
- pillow
- opencv
- visdom (optional)
- tensorboardX (optional)

## Architecture
```
Detection
├── cfg
├── data
├── dataset
├── engine
├── littlenet
│   ├── loss
│   ├── models
│   ├── network
│   │   ├── backbone
│   │   ├── head
│   │   ├── layer
│   │   ├── neck
│   │   ├── roi
│   │   └── plugins
│   └── utils
├── utils
└── weights
```

## Features
  Include YOLOv3, SSD, SSDLite, RetinaNet, Faster RCNN, Cascade RCNN

## Reference
- [OneStageDet](https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

---
## Usage
First compile related components
```
cd littlenet/utils
make
```
### Dataset
Cityscapes Json type
```
.
├── annotations     json files 4 train, val, test annotation
└── leftImg8bit     images
    ├── test
    ├── train
    └── val
```
VOC type
```
.
├── images
├── annotations_test.pkl
└── annotations_train.pkl
```

### Training
```bash
mkdir data
mkdir backup
mkdir -p log/visdom
```
`data/` where to put dataset, while the training weights backup save in `backup/` <br>

train.py
```
usage: train.py [-h] [-d DATA] -m MODEL [-w WEIGHT] [-v] [-b] [-s SEED]
                [--val]

Train a model

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Dataset to use
  -m MODEL, --md MODEL  Model config file
  -w WEIGHT, --weight WEIGHT
                        Initial weight file
  -v, --visual          Use Visdom to log
  -b, --board           Use tensorboard to log
  -s SEED, --seed SEED  Set random seed
  --val                 Val when training

```

config file in `./cfg`

### Evaluation
test.py
```
usage: test.py [-h] [-d DATA] -m MODEL -w WEIGHT [-v]

Test a model

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Dataset to use
  -m MODEL, --md MODEL  Model config file
  -w WEIGHT, --weight WEIGHT
                        Weight file
  -v, --vis             Visualize results

```

## TODO List
- [ ] Multi GPUs training
- [ ] Visualization
- [ ] TorchScript Convert
- [ ] GN