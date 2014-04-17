
from ..dataset import Dataset
from .forward import ForwardModel 

class NNForwardModel(ForwardModel):
    """Nearest Neighbors Forward Model"""
    
    name = 'NN'
    desc = 'Nearest Neighbors'
    
    def predict_y(self, xq, **kwargs):
        """Provide an prediction of xq in the output space

        @param xq  an array of float of length dim_x
        @return    predicted y as np.array of float
        """
        dists, indexes = self.dataset.nn_x(xq, k = 1)
        return self.dataset.get_y(indexes[0])
        