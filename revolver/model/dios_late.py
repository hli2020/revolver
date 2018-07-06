import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .backbone import vgg16
from .fcn import Interpolator, Downsampler


class dios_late(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # downsampling for annotation mask
        self.anno_enc = Downsampler(1, 32, odd=False)

        # Make fully conv encoder
        backbone = vgg16(is_caffe=True)
        for k in list(backbone._modules)[-6:]:
            del backbone._modules[k]
        feat_dim = 256
        fc6 = [('fc6', nn.Conv2d(512, feat_dim, 7)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc6_drop', nn.Dropout2d(p=0.5))]
        backbone._modules.update(fc6)
        self.encoder = backbone

        # Make head
        head = [('fc7', nn.Conv2d(feat_dim*3, feat_dim*3, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5)),
            ('score', nn.Conv2d(feat_dim*3, num_classes, 1))]
        self.head = nn.Sequential(OrderedDict(head))

        # FC6 and FC7 should be init with random Gaussian weights
        # Score layer should be zero
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

    def forward(self, im, anno):
        h, w = im.size()[-2:]

        # Extract image features
        im = self.encoder(im)

        # Pre-process annotations and downsample them
        anno = F.pad(anno, (0, 31, 0, 31), 'constant', 0)
        pos = Variable(anno[:, :1, ...].data.clone())
        neg = Variable(anno[:, 1:, ...].data.clone())
        pos_anno = self.anno_enc(pos)
        neg_anno = self.anno_enc(neg)
        pos_anno = pos_anno / (1e-6 + torch.sum(pos_anno.view(-1), dim=0))
        neg_anno = neg_anno / (1e-6 + torch.sum(neg_anno.view(-1), dim=0))

        # Mask image features
        im_feats = self.mask_feat(im, pos_anno, scale=False)
        pos_feats = self.mask_feat(im, pos_anno)
        neg_feats = self.mask_feat(im, neg_anno)

        # Concatenate masked features
        feat = torch.cat([im_feats, pos_feats, neg_feats], dim=1)

        # Final conv, score, upsample to make prediction
        scores = self.head(feat)
        upscores = self.decoder(scores)
        upscores = upscores[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return upscores

    def mask_feat(self, x, mask, scale=True):
        """
        Align spatial coordinates of feature and mask, crop feature, and
        multiply by mask if scale is True.

        Expect feature and mask to be N x C x H x W
        """
        # With input pad 81, fc6 crop offset is 0, so align upper lefts
        x_size, mask_size = x.size(), mask.size()
        if x_size[-2:] != mask_size[-2:]:
            raise ValueError("Shape mismatch. Feature is {}, but mask is {}".format(x_size, mask_size))
        m_dim = mask_size[-2:]
        x = x[:, :, :m_dim[0], :m_dim[1]]
        if scale:
            x = x * mask
        return x
