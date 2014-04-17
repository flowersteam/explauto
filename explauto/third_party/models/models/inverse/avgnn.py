"""Average of Nearest Neighbors inverse model.
In such a model, the crucial part is choosing the good neighborhood.

Notes : There should be no particular reason why the NN forward model's k
        should be the same as the NN inverse model's k, but it's not such
        a bad heuristic. Could probably be improved.
"""

import random
import numpy as np

from ....toolbox import toolbox
from ..forward.avgnn import AverageNNForwardModel
from . import inverse

relax = 0.0 # no relaxation

class AverageNNInverseModel(inverse.InverseModel):
    """Average Nearest Neighbors Inverse Model"""

    name = 'AvgNN'
    desc = 'Averaged Nearest Neighbors'

    def __init__(self, dim_x, dim_y, k = None, **kwargs):
        """
        @param k  the number of neighbors to consider for averaging
        """
        inverse.InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.fmodel = AverageNNForwardModel(dim_x, dim_y, k = k, **kwargs)
        self.k      = k or 3*dim_y

    def infer_x(self, y, k = None):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        @param k  how many neighbors to consider for the average
                  this value override the class provided one on a per
                  method call basis.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        k = k or self.k

        x_guess = self._guess_x(y, k = k)[0]
        dists, index = self.fmodel.dataset.nn_x(x_guess, k = k)
        return [np.average([self.fmodel.dataset.get_x(idx) for idx in index], axis = 0)]

    def _guess_x(self, y_desired, **kwargs):
        """Choose the relevant neighborhood on which to average, based on the minimum
        spread of the corresponding neighborhood in S.
        for each (x, y) with y neighbor of y_desired,
            1. find the neighborhood of x, (xi, yi)_k.
            2. compute the standart deviation of the error between yi and y_desired.
            3. select the neighborhood of minimum standart deviation

        TODO : Implement another method taking the spread in M too.
        """
        k = kwargs.get('k', self.k)
        dists, indexes = self.fmodel.dataset.nn_y(y_desired, k = k)
        min_std, min_xi = float('inf'), None
        for i in indexes:
            xi = self.fmodel.dataset.get_x(i)
            dists_xi, indexes_xi = self.fmodel.dataset.nn_x(xi, k = k)
            std_xi = np.std([toolbox.dist(self.fmodel.dataset.get_y(j), y_desired) for j in indexes_xi])
            if std_xi < min_std:
                min_std, min_xi = std_xi, xi
        return [min_xi]
