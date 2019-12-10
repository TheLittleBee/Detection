import torch
import torch.onnx.symbolic
from torch.autograd import Variable
import argparse

from utils.config import HyperParams
from littlenet import models


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('-d', '--data', dest='data', type=str, default='LAB', help='Dataset to use')
    parser.add_argument('-m', '--md', dest='model', type=str, required=True, help='Model config file')
    parser.add_argument('-w', '--weight', dest='weight', type=str, required=True, help='Weight file')
    parser.add_argument('--onnx', dest='onnx', action='store_true', help='export as onnx or torchscript')
    args = parser.parse_args()
    return args

# @torch.onnx.symbolic.parse_args('v', 'is')
# def upsample_nearest2d(g, input, output_size):
#     height_scale = float(output_size[-2]) / input.type().sizes()[-2]
#     width_scale = float(output_size[-1]) / input.type().sizes()[-1]
#     return g.op("Upsample", input,
#                 scales_f=(1, 1, height_scale, width_scale),
#                 mode_s="nearest")
#
# torch.onnx.symbolic.upsample_nearest2d = upsample_nearest2d

def main():
    args = parse_args()
    print(args)

    hyper = HyperParams(args, 2)

    model_name = hyper.model_name
    model_cfg = hyper.model_cfg
    model_type = model_cfg.pop('type')
    assert model_type in models.__dict__
    bg = model_cfg.pop('background')
    model_cfg['weights'] = hyper.weights
    model_cfg['head']['num_classes'] = hyper.n_cls + 1 if bg else hyper.n_cls
    net = models.__dict__[model_type](**model_cfg)

    net.eval()
    x_input = torch.rand(1, 3, hyper.input_size[1], hyper.input_size[0], dtype=torch.float)
    if args.onnx:
        net.export(x_input, 'result/%s_%s_export.onnx' % (model_name, args.data))
        torch.onnx.export(net, x_input, 'result/%s_%s.onnx' % (model_name,args.data), verbose=True)
    else:
        # net.exporting = True
        script = torch.jit.trace(net, x_input)
        script.save('result/%s_%s.pt' % (model_name, args.data))

from littlenet.utils.bbox import xywh2xyxy
class Test(torch.nn.Module):
    def __init__(self, anchors):
        self.anchors = anchors
        super(Test, self).__init__()

    def forward(self, output):
        nB,C,nH,nW = output.shape
        nA = 2
        nC = C//nA - 5

        output = output.view(nB, nA, -1, nH * nW)
        # coordxy = output[:, :, :2].sigmoid()  # tx,ty
        # coordwh = output[:, :, 2:4]  # tw,th
        # coord = torch.cat((coordxy, coordwh), 2)
        # conf = output[:, :, 4].sigmoid()
        # if nC > 1:
        #     cls = output[:, :, 5:].contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)
        #
        # lin_x = torch.arange(nW, dtype=torch.float).repeat(nH, 1).view(nH * nW)
        # lin_y = torch.arange(nH, dtype=torch.float).repeat(nW, 1).t().contiguous().view(nH * nW)
        # anchor_w = self.anchors[:, 0].view(nA, 1)
        # anchor_h = self.anchors[:, 1].view(nA, 1)
        #
        # pred_boxesx = (coord[:, :, 0] + lin_x).view(-1)
        # pred_boxesy = (coord[:, :, 1] + lin_y).view(-1)
        # pred_boxesw = (coord[:, :, 2].exp() * anchor_w).view(-1)
        # pred_boxesh = (coord[:, :, 3].exp() * anchor_h).view(-1)
        # pred_boxes = torch.stack((pred_boxesx, pred_boxesy, pred_boxesw, pred_boxesh), 1)
        # # for inference
        # pred_boxes = xywh2xyxy(pred_boxes) * 32
        # score = torch.nn.functional.softmax(cls, 1).view(nB, -1, nC) * conf.view(nB, -1, 1)
        # idx = torch.arange(nC).repeat(nB, score.size(1)).float()
        # pred_boxes = pred_boxes.repeat(1, nC)
        # # # x1,y1,x2,y2,score,idx     repeat cls agnostic boxes
        # det = torch.cat((pred_boxes.view(nB, -1, 4), score.view(nB, -1, 1), idx.view(nB, -1, 1)), -1)
        return output


if __name__ == '__main__':
    # anchors = torch.Tensor([[300,300],[200,200]])
    # md = Test()
    # x = torch.rand(1, 3, 7, 7)
    # torch.onnx.export(md, x, 'result/test.onnx', verbose=True, export_params=True)
    # script = torch.jit.trace(md, x)
    # print(md(x))
    # print(script(x))
    main()
