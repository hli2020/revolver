import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator


class dios_early_lite(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = vgg16(is_caffe=True)
        for k in list(backbone._modules)[-6:]:
            del backbone._modules[k]
        feat_dim = 256
        fc6 = [('fc6', nn.Conv2d(512, feat_dim, 7)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc6_drop', nn.Dropout2d(p=0.5))]
        backbone._modules.update(fc6)

        # FC6 and FC7 should be init with random Gaussian weights
        # Score layer should be zero
        for n, m in self.named_modules():
            if 'fc' in n and isinstance(n, nn.Conv2d):
                m.weight.data.normal_(0, .001)
            elif 'score' in n:
                m.weight.data.fill_(0.)
                m.bias.data.fill_(0.)
        # Modify conv1_1 to have 5 input channels
        # Init the weights in the new channels to the channel-wise mean
        # of the pre-trained conv1_1 weights
        old_conv1 = backbone.conv1_1.weight.data
        mean_conv1 = torch.mean(old_conv1, dim=1, keepdim=True)
        new_conv1 = nn.Conv2d(5, old_conv1.size(0), kernel_size=old_conv1.size(2), stride=1, padding=1)
        new_conv1.weight.data = torch.cat([old_conv1, mean_conv1, mean_conv1], dim=1)
        new_conv1.bias.data = backbone.conv1_1.bias.data
        backbone.conv1_1 = new_conv1
        self.encoder = backbone

        # classifier head
        head = [('fc7', nn.Conv2d(feat_dim, feat_dim, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5)),
            ('score', nn.Conv2d(feat_dim, num_classes, 1))]
        self.head = nn.Sequential(OrderedDict(head))

        # init fc[6-7] w/ gaussian
        # init score w/ zero
        for n, m in self.named_modules():
            if 'fc' in n and isinstance(n, nn.Conv2d):
                m.weight.data.normal_(0, .001)
            elif 'score' in n:
                m.weight.data.fill_(0.)
                m.bias.data.fill_(0.)

        # bilinear interpolation for upsampling
        self.decoder = Interpolator(num_classes, 32, odd=False)
        # align output to input: see
        # https://github.com/BVLC/caffe/blob/master/python/caffe/coord_map.py
        self.encoder[0].padding = (81, 81)
        self.crop = 0

    def forward(self, x, anno):
        x = torch.cat((x, anno), dim=1)
        h, w = x.size()[-2:]
        x = self.encoder(x)
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x
