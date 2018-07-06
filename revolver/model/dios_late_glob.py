import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .dios_late import dios_late


class dios_late_glob(dios_late):

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

        # Global pool pos and neg features
        pos_glob = torch.sum(pos_feats.view(pos_feats.size(0), pos_feats.size(1), -1), dim=2) #C
        pos_glob = pos_glob[..., None, None] # 1 x C x 1 x 1
        neg_glob = torch.sum(neg_feats.view(neg_feats.size(0), neg_feats.size(1), -1), dim=2)
        neg_glob = neg_glob[..., None, None]

        # Tile the pooled features across the image feature
        pos_glob = pos_glob.repeat(1,1,im_feats.size(2), im_feats.size(3))
        neg_glob = neg_glob.repeat(1,1,im_feats.size(2), im_feats.size(3))
        feat = torch.cat([im_feats, pos_glob, neg_glob], dim=1)

        # Final conv, score, upsample to make prediction
        scores = self.head(feat)
        upscores = self.decoder(scores)
        upscores = upscores[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return upscores

