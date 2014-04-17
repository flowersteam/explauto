"""Average of Nearest Neighbors inverse model.
In such a model, the crucial part is choosing the good neighborhood.

Notes : There should be no particular reason why the NN forward model's k
        should be the same as the NN inverse model's k, but it's not such
        a bad heuristic. Could probably be improved.
"""

import math
import random
import numpy as np

from ....toolbox import toolbox
from ..forward import AverageNNForwardModel
from . import inverse

relax = 0.0 # no relaxation

class WeightedNNInverseModel(inverse.InverseModel):
    """Weighted Nearest Neighbors Inverse Model"""

    name = 'WNN'
    desc = 'Weighted Nearest Neighbors'

    def __init__(self, dim_x, dim_y, sigma = 1.0, k = None, **kwargs):
        """
        @param k      the number of neighbors to consider for averaging
        @param sigma  for the moment, default sigma for forward model is the same as
                      the one of the inverse model. Not ideal. #FIXME
        """
        self.k      = k or 3*dim_x
        self.fmodel = AverageNNForwardModel(dim_x, dim_y, sigma = sigma, k = self.k, **kwargs)
        self.sigma  = sigma

    def infer_x(self, y, sigma = None, k = None, **kwargs):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        @param k  how many neighbors to consider for the average
                  this value override the class provided one on a per
                  method call basis.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        k = k or self.k
        sigma = sigma or self.sigma

        x_guess = self._guess_x(y, k = k)[0]
        dists, index = self.fmodel.dataset.nn_x(x_guess, k = k)
        w = self._weights(index, dists, sigma*sigma,  y)
        return [np.sum([wi*self.fmodel.dataset.get_x(idx)
                for wi, idx in zip(w, index)], axis = 0)]

    def _weights(self, index, dists, sigma_sq, y_desired):

        dists = [toolbox.dist(self.fmodel.dataset.get_y(idx), y_desired)
                 for idx in index] # could be optimized

        w = np.fromiter((toolbox.gaussian_kernel(d/self.fmodel.dim_y, sigma_sq)
                         for d in dists), np.float)

        # We eliminate the outliers # TODO : actually reduce w and index
        wsum = w.sum()
        if wsum == 0:
            return 1.0/len(dists)*np.ones((len(dists),))
        else:
            eps = wsum * 1e-15 / self.fmodel.dim_y
            return np.fromiter((w_i/wsum if w_i > eps else 0.0 for w_i in w), np.float)


class ESWNNInverseModel(WeightedNNInverseModel):
    """ES-WNN : WNN with estimated sigma, on a query basis, as the mean distance."""

    name = 'ES-WNN'

    def _weights(self, index, dists, sigma_sq, y_desired):
        sigma_sq=(dists**2).sum()/len(dists)
        return WeightedNNInverseModel._weights(self, index, dists, sigma_sq, y_desired)
