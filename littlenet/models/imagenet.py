from littlenet.network import backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class Classfy_Head(nn.Module):
    def __init__(self, inp, mid, oup):
        super(Classfy_Head, self).__init__()
        layer = [
            OrderedDict([
                ('Conv1', nn.Sequential(
                    nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(mid),
                    nn.ReLU(inplace=True),
                )),
            ]),
            OrderedDict([
                ('Conv2', nn.Conv2d(mid, oup, 1, 1, 0, bias=True))
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer])

    def forward(self, middle_feats):
        x = self.layers[0](middle_feats[0])
        x = F.avg_pool2d(x, kernel_size=(7, 7), stride=(1, 1))
        return self.layers[1](x)


class ImageNet_C(nn.Module):
    def __init__(self):
        super(ImageNet_C, self).__init__()
        self.backbone = backbone.vgg16bn()
        # self.head = Classfy_Head(320, 1280, 1001)

    def forward(self, x):
        f = self.backbone(x)
        # o = self.head(f)
        # return torch.squeeze(o)
        return f

    def load_weights(self, file):
        fp = open(file, 'rb')
        # header = np.fromfile(fp, dtype=np.int32, count=5)
        # self.header = header
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()

        ptr = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                num_w = m.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.weight)
                m.weight.data.copy_(conv_w)
                ptr += num_w
                if m.bias is not None:
                    num_b = m.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(m.bias)
                    m.bias.data.copy_(conv_b)
                    ptr += num_b
            if isinstance(m, nn.BatchNorm2d):
                num_w = m.weight.numel()
                bn_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.weight)
                m.weight.data.copy_(bn_w)
                ptr += num_w
                bn_b = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.bias)
                m.bias.data.copy_(bn_b)
                ptr += num_w
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.running_mean)
                m.running_mean.data.copy_(bn_rm)
                ptr += num_w
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(m.running_var)
                m.running_var.data.copy_(bn_rv)
                ptr += num_w
            if ptr == len(weights): print('Load success')
            if ptr >= len(weights): break

    def partial(self, path):
        torch.save(self.backbone.state_dict(), path)

    def save_weights(self, path):
        fp = open(path, 'wb')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.cpu().numpy().tofile(fp)
                if m.bias is not None:
                    m.bias.data.cpu().numpy().tofile(fp)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.cpu().numpy().tofile(fp)
                m.bias.data.cpu().numpy().tofile(fp)
                m.running_mean.data.cpu().numpy().tofile(fp)
                m.running_var.data.cpu().numpy().tofile(fp)
        fp.close()


if __name__ == '__main__':
    from PIL import Image

    model = ImageNet_C()
    model.load_weights('/home/littlebee/code/pytorch/Detection/weights/vgg16bn.weights')
    # model.backbone.load_state_dict(torch.load('/home/littlebee/code/pytorch/Detection/weights/darknet53.features'))
    model.eval()
    model.partial('/home/littlebee/code/pytorch/Detection/weights/vgg16bn.features')
    # model.save_weights('/home/littlebee/code/pytorch/Detection/weights/darknet53.weights')

    # img = Image.open('/home/littlebee/code/pytorch/Detection/dataset/dog.jpg')
    # img=img.resize((224,224))
    # x=np.transpose(np.array(img),(2,0,1))/255.
    # X = torch.from_numpy(x[np.newaxis]).float()
    # print(X.size())
    #
    # predict = model(X)
    # print(predict.size())
    # print(torch.argmax(predict))
