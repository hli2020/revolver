import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator


class fcn32s(nn.Module):
    """
    FCN-32s: fully convolutional network with VGG-16 backbone.
    """

    def __init__(self, num_classes):
        super().__init__()
        # pre-trained features
        self.encoder = vgg16(is_caffe=True)

        # classifier head
        self.head = nn.Conv2d(4096, num_classes, 1)
        self.head.weight.data.fill_(0.)
        self.head.bias.data.fill_(0.)

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
