import numpy as np

from ..dataset import Dataset
from .forward import ForwardModel 

class AverageNNForwardModel(ForwardModel):
    """Forward model based on averaging the k nearest neighbors"""

    name = 'AvgNN'
    desc = 'Averaged Nearest Neighbors'

    def __init__(self, dim_x, dim_y, k = None, **kwargs):
        """Create the forward model

        @param k        how many neighbors to consider for the average
        """
        self.k = k or 3*dim_x
        ForwardModel.__init__(self, dim_x, dim_y, k = self.k, **kwargs)

    def predict_y(self, xq, k = None, **kwargs):
        """Provide an prediction of xq in the output space

        @param xq  an array of float of length dim_x
        @param k   how many neighbors to consider for the average
                   this value override the class provided one on a per 
                   method call basis.
        @return    predicted y as np.array of float
        """
        k = k or self.k
        dists, indexes = self.dataset.nn_x(xq, k = k)
        return np.average([self.dataset.get_y(idx) for idx in indexes], axis = 0)
