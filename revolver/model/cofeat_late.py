import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .dios_late import dios_late


class cofeat_late(dios_late):

    def forward(self, qry, supp):
        # query
        h, w = qry.size()[-2:]
        qry = self.encoder(qry)

        # encode support images
        supp_feats = [self.encoder(im) for im, _ in supp]

        # cast annotations into masks for feature maps
        pos_annos, neg_annos = [], []
        for _, anno in supp:
            anno = F.pad(anno, (0, 31, 0, 31), 'constant', 0)
            pos = Variable(anno[:, :1, ...].data.clone())
            neg = Variable(anno[:, 1:, ...].data.clone())
            pos_anno = self.anno_enc(pos)
            neg_anno = self.anno_enc(neg)
            pos_anno = pos_anno / (1e-6 + torch.sum(pos_anno.view(-1), dim=0))
            neg_anno = neg_anno / (1e-6 + torch.sum(neg_anno.view(-1), dim=0))
            pos_annos.append(pos_anno)
            neg_annos.append(neg_anno)

        # mask support by annotations
        pos_feats = [self.mask_feat(f, a) for f, a in zip(supp_feats, pos_annos)]
        neg_feats = [self.mask_feat(f, a) for f, a in zip(supp_feats, neg_annos)]

        # global pool support +/- features and tile across query feature
        pos_vec = torch.cat([f.view(1, f.size(1), -1) for f in pos_feats], dim=2)
        neg_vec = torch.cat([f.view(1, f.size(1), -1) for f in neg_feats], dim=2)
        pos_glob = torch.sum(pos_vec, dim=2)  # 1 x C
        neg_glob = torch.sum(neg_vec, dim=2)
        pos_glob = pos_glob[..., None, None]  # 1 x C x 1 x 1
        neg_glob = neg_glob[..., None, None]
        # normalize by support size (mask is normalized by no. annotations)
        pos_glob = pos_glob.div_(len(supp))
        neg_glob = neg_glob.div_(len(supp))

        # Tile the pooled features across the image feature
        pos_glob = pos_glob.repeat(1, 1, qry.size(2), qry.size(3))
        neg_glob = neg_glob.repeat(1, 1, qry.size(2), qry.size(3))
        x = torch.cat([qry, pos_glob, neg_glob], dim=1)

        # inference from combined feature
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x

