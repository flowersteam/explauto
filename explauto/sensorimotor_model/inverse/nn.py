

import numpy as np

from .. import forward
from . import inverse


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
