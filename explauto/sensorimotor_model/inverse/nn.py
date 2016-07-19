

import numpy as np

from .. import forward
from . import inverse
from explauto.utils import gaussian_kernel


class NNInverseModel(inverse.InverseModel):
    """Nearest Neighbor Inverse Model"""

    name = 'NN'
    desc = 'Nearest Neighbors'

    def __init__(self, dim_x, dim_y, fmodel, **kwargs):
        """
        @param k  the number of neighbors to consider for averaging
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        inverse.InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.fmodel = fmodel
        self.k = fmodel.k

    def infer_x(self, y):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        if len(self.fmodel.dataset) == 0:
            return [[0.0]*self.dim_x]
        else:
            _, index = self.fmodel.dataset.nn_y(y, k=1)
            return [self.fmodel.dataset.get_x(index[0])]

    def infer_dm(self, m, s, ds):
        return self.infer_dims(m, np.hstack((s, ds)), range(len(m)), range(self.dim_x, self.dim_x + self.dim_y), range(len(m), self.dim_x))
        
        
    def infer_dims(self, x, y, dims_x, dims_y, dims_out):
        """Infer probable output from input x, y
        """
        assert len(x) == len(dims_x)
        assert len(y) == len(dims_y)
        if len(self.fmodel.dataset) == 0:
            return [0.0]*self.dim_out
        else:
            _, index = self.fmodel.dataset.nn_dims(x, y, dims_x, dims_y, k=1)
            return self.fmodel.dataset.get_dims(index[0], dims=dims_out)


class NSNNInverseModel(inverse.InverseModel):
    """Non-Stationary Nearest Neighbor Inverse Model"""

    name = 'NSNN'
    desc = 'Non-Stationary Nearest Neighbors'

    def __init__(self, dim_x, dim_y, fmodel, sigma=1.0, sigma_t=100, **kwargs):
        """
        @param k  the number of neighbors to consider for averaging
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        inverse.InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.fmodel = fmodel
        self.k = fmodel.k
        self.sigma_sq = sigma*sigma
        self.sigma_t_sq = sigma_t*sigma_t

    def infer_x(self, y):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        if len(self.fmodel.dataset) == 0:
            return [[0.0]*self.dim_x]
        else:
            dists, index = self.fmodel.dataset.nn_y(y, k=self.k)
            w = self._weights(dists, index)
            idx = index[np.argmax(w)]
            return [self.fmodel.dataset.get_x(idx)]

    def _weights(self, dists, index):
        w = np.fromiter((gaussian_kernel(d, self.sigma_sq)
                         for d in dists), np.float, len(dists))

        # Weight by timestamp of samples to forget old values
        max_index = max(index)        
        wt = np.fromiter((gaussian_kernel(max_index - idx, self.sigma_t_sq)
                         for idx in index), np.float, len(dists))
        w = w * wt    
        wsum = w.sum()
        if wsum == 0:
            return 1.0/len(dists)*np.ones((len(dists),))
        else:
            eps = wsum * 1e-10 / self.dim_x
            return np.fromiter((w_i/wsum if w_i > eps else 0.0 for w_i in w), np.float)
