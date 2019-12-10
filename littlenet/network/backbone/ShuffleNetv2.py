import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..layer.shufflenet import InvertedResidual

__all__ = ['shufflenetv2_x0_5', 'shufflenetv2']

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        stage1 = self.conv1(x)
        stage2 = self.maxpool(stage1)
        stage3 = self.stage2(stage2)
        stage4 = self.stage3(stage3)
        stage5 = self.stage4(stage4)
        # stage5 = self.conv5(stage5)

        return stage5, stage4, stage3


def shufflenetv2(pretrained=True):
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['shufflenetv2']), strict=False)
    return model


def shufflenetv2_x0_5(pretrained=True):
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['shufflenetv2_x0.5']), strict=False)
    return model
