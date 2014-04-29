# -*- coding: utf-8 -*-

"""
Weighted Average of Nearest Neighbors Forward Model.
"""

import math

import numpy as np
#np.set_printoptions(precision=6, linewidth=300)

from ....toolbox import toolbox
from ..dataset import Dataset
from .forward import ForwardModel

class WeightedNNForwardModel(ForwardModel):
    """Weighted Nearest Neighbors Forward Model"""

    name = 'WNN'
    desc = 'Weighted Nearest Neighbors'

    def __init__(self, dim_x, dim_y, sigma = 1.0, k = None, **kwargs):
        """Create the forward model

        @param sigma    sigma for the guassian distance.
        @param k        the number of nearest neighbors to consider for regression.
        """
        self.k         = k or 3*dim_x
        ForwardModel.__init__(self, dim_x, dim_y, sigma = sigma, k = self.k, **kwargs)
        self.sigma     = sigma

    def predict_y(self, xq, sigma = None, k = None, **kwargs):
        """Provide an prediction of xq in the output space

        @param xq  an array of float of length dim_x
        """
        sigma = sigma or self.sigma
        k = k or self.k
        dists, index = self.dataset.nn_x(xq, k = k)
        w = self._weights(dists, sigma*sigma)
        return np.sum([wi*self.dataset.get_y(idx) for wi, idx in zip(w, index)], axis = 0)

    def _weights(self, dists, sigma_sq):

        w = np.fromiter((toolbox.gaussian_kernel(d/self.dim_x, sigma_sq)
                         for d in dists), np.float)

        # We eliminate the outliers # TODO : actually reduce w and index
        wsum = w.sum()
        if wsum == 0:
            return 1.0/len(dists)*np.ones((len(dists),))
        else:
            eps = wsum * 1e-15 / self.dim_x
            return np.fromiter((w_i/wsum if w_i > eps else 0.0 for w_i in w), np.float)

class ESWNNForwardModel(WeightedNNForwardModel):
    """ES-WNN : WNN with estimated sigma, on a query basis, as the mean distance."""

    def __init__(self, *args, **kwargs):
        WeightedNNForwardModel.__init__(self, *args, **kwargs)
        self.name = 'ES-WNN'

    def _weights(self, dists, sigma_sq):
        sigma_sq=(dists**2).sum()/len(dists)
        return WeightedNNForwardModel._weights(self, dists, sigma_sq)
