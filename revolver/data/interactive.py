import numpy as np

from torch.utils.data import Dataset

from .util import Wrapper


class InteractiveSeg(Wrapper, Dataset):
    """
    Construct inputs (image and sparse annotations) and targets
    (dense annotations) for an interactive segmentation model.

    Args:
        dense_ds: the `Dataset` to load dense labels as targets
        sparse_ds: the `Dataset` to load sparse labels as input annotations

    Note that this class assumes the two input datasets contain
    the same data in the same order.
    """

    def __init__(self, dense_ds, sparse_ds):
        super().__init__(dense_ds)
        self.dense_ds = dense_ds
        self.sparse_ds = sparse_ds

    def __getitem__(self, idx):
        # load regular image + target
        im, target, aux = self.dense_ds[idx]
        # load sparse input annotations
        _, anno, sparse_aux = self.sparse_ds[idx]
        aux.update(sparse_aux)
        # (float is necessary downstream for interpolation and such)
        pos = (anno == 1).astype(np.float32)
        neg = (anno == 0).astype(np.float32)
        stacked_anno = np.concatenate((pos[None, ...], neg[None, ...]), axis=0)
        return im, stacked_anno, target, aux

    def __len__(self):
        return len(self.dense_ds)
