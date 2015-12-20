
import numpy as np

from .forward import ForwardModel
from explauto.utils import gaussian_kernel


class NNForwardModel(ForwardModel):
    """Nearest Neighbors Forward Model"""

    name = 'NN'
    desc = 'Nearest Neighbors'

    def __init__(self, dim_x, dim_y, sigma=1.0, **kwargs):
        """Create the forward model

        @param dim_x    the input dimension
        @param dim_y    the output dimension
        @param sigma    sigma for the guassian distance.
        @param sigma_t  sigma for the time distance.
        @param nn       the number of nearest neighbors to consider for regression.
        """
        self.k = 1
        ForwardModel.__init__(self, dim_x, dim_y, sigma=sigma, **kwargs)
        
    def predict_y(self, xq, **kwargs):
        """Provide an prediction of xq in the output space

        @param xq  an array of float of length dim_x
        @return    predicted y as np.array of float
        """
        _, indexes = self.dataset.nn_x(xq, k = 1)
        return self.dataset.get_y(indexes[0])
        
    def predict_given_context(self, x, c, c_dims):
        """Provide a prediction of x with context c on dimensions c_dims in the output space being S - c_dims

        @param x  an array of float of length dim_x
        @return    predicted y as np.array of float
        """
        assert len(c) == len(c_dims)
        _, index = self.dataset.nn_dims(x, c, range(self.dim_x), list(np.array(c_dims) + self.dim_x), k=1)
        return self.dataset.get_y(index[0])


class NSNNForwardModel(ForwardModel):
    """Non-Stationary Nearest Neighbors Forward Model
        TODO: cpp implementation of NSNN                            """

    name = 'NSNN'
    desc = 'Non-Stationary Nearest Neighbors'

    def __init__(self, dim_x, dim_y, sigma=1.0, sigma_t=100, k=20, **kwargs):
        """Create the forward model

        @param dim_x    the input dimension
        @param dim_y    the output dimension
        @param sigma    sigma for the guassian distance.
        @param sigma_t  sigma for the time distance.
        @param nn       the number of nearest neighbors to consider for regression.
        """
        self.k = k
        ForwardModel.__init__(self, dim_x, dim_y, sigma = sigma, k = self.k, **kwargs)
        self.sigma_sq = sigma*sigma
        self.sigma_t_sq = sigma_t*sigma_t

    @property
    def sigma(self):
        return self.conf['sigma']

    @sigma.setter
    def sigma(self, sigma):
        self.sigma_sq = sigma*sigma
        self.conf['sigma'] = sigma

    #@profile
    def predict_y(self, xq, sigma=None, k = None):
        """Provide an prediction of xq in the output space

        @param xq  an array of float of length dim_x
        @param estimated_sigma  if False (default), sigma_sq=self.sigma_sq, else it is estimated from the neighbor distances in self._weights(.)
        """
        #print(len(xq) , self.dataset.dim_x)
        assert len(xq) == self.dataset.dim_x
        sigma_sq = self.sigma_sq if sigma is None else sigma*sigma
        k = k or self.k

        dists, index = self.dataset.nn_x(xq, k = k)
        #print(list(index))

        w = self._weights(dists, index, sigma_sq)
        idx = index[np.argmax(w)]
        return self.dataset.get_y(idx)

    def _weights(self, dists, index, sigma_sq):
        w = np.fromiter((gaussian_kernel(d, sigma_sq)
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
