import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator


class fcn32s_lite(nn.Module):
    """
    FCN-32s-lite: fully convolutional network with VGG-16 backbone,
    light-headed edition with only 256 channels after conv5.
    """

    def __init__(self, num_classes):
        super().__init__()
        # encoder: decapitated VGG16 w/ channel dim. 256
        backbone = vgg16(is_caffe=True)
        for k in list(backbone._modules)[-6:]:
            del backbone._modules[k]
        feat_dim = 256  # lite-headed: 256 vs. regular 4096
        fc6 = [('fc6', nn.Conv2d(512, feat_dim, 7)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc6_drop', nn.Dropout2d(p=0.5))]
        backbone._modules.update(fc6)
        self.encoder = backbone

        # classifier head
        head = [('fc7', nn.Conv2d(feat_dim, feat_dim, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5)),
            ('score', nn.Conv2d(feat_dim, num_classes, 1))]
        self.head = nn.Sequential(OrderedDict(head))

        # weight init:
        # - fc6, fc7 are random Gaussian
        # - score is zero
        for n, m in self.named_modules():
            if 'fc' in n and isinstance(n, nn.Conv2d):
                m.weight.data.normal_(0, .001)
                m.bias.data.fill_(0.)
            elif 'score' in n:
                m.weight.data.fill_(0.)
                m.bias.data.fill_(0.)

        # bilinear interpolation for upsampling
        self.decoder = Interpolator(num_classes, 32, odd=False)
        # align output to input: see
        # https://github.com/BVLC/caffe/blob/master/python/caffe/coord_map.py
        self.encoder[0].padding = (81, 81)
        self.crop = 0

    def forward(self, x):
        h, w = x.size()[-2:]
        x = self.encoder(x)
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x
