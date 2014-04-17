
import random
import numpy as np

from ....toolbox import toolbox
from ..forward import NNForwardModel
from . import inverse

relax = 0.0 # no relaxation

class NNInverseModel(inverse.InverseModel):
    """Nearest Neighbor Inverse Model"""

    name = 'NN'
    desc = 'Nearest Neighbors'

    def __init__(self, dim_x, dim_y, **kwargs):
        """
        @param k  the number of neighbors to consider for averaging
        """
        inverse.InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.fmodel = NNForwardModel(dim_x, dim_y, **kwargs)

    def infer_x(self, y):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        dists, index = self.fmodel.dataset.nn_y(y, k = 1)
        return [self.fmodel.dataset.get_x(index[0])]
