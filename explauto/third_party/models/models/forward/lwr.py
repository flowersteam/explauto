# -*- coding: utf-8 -*-

"""

Locally Weigthed Regression (LWR) python implementation.

References :
    1. C. G. Atkeson, A. W. Moore, S. Schaal, "Locally Weighted Learning for Control",
         "Springer Netherlands", 75-117, vol 11, issue 1, 1997/02, 10.1023/A:1006511328852
    For a video lecture :
    2. http://www.cosmolearning.com/video-lectures/locally-weighted-regression-probabilistic-interpretation-logistic-regression/

Pseudo Code :

    Input X matrix of inputs:  X[k] [i] = i’th component of k’th input point.
    Input Y matrix of outputs: Y[k] = k’th output value.
    Input xq = query input.    Input kwidth.

    WXTWX = empty (D+1) x (D+1) matrix
    WXTWY = empty (D+1) x 1     matrix

    for k in range(N):

        /* Compute weight of kth point  */
        wk = weight_function( distance( xq , X[k] ) / kwidth )

        /* Add to (WX) ^T (WX) matrix */
        for ( i = 0 ; i <= D ; i = i + 1 )
            for ( j = 0 ; j <= D ; j = j + 1 )
                if ( i == 0 )
                    xki = 1 else xki = X[k] [i]
                if ( j == 0 )
                    xkj = 1 else xkj = X[k] [j]
                WXTWX [i] [j] = WXTWX [i] [j] + wk * wk * xki * xkj

        /*  Add to (WX) ^T (WY) vector */
        for ( i = 0 ; i <= D ; i = i + 1 )
            if ( i == 0 )
                xki = 1 else xki = X[k] [i]
            WXTWY [i] = WXTWY [i] + wk * wk * xki * Y[k]

    /* Compute the local beta.  Call your favorite linear equation solver.
       Recommend Cholesky Decomposition for speed.
       Recommend Singular Val Decomp for Robustness. */

    Beta = (WXTWX)^{-1}(WXTWY)

    Output ypredict = beta[0] + beta[1]*xq[1] + beta[2]*xq[2] + … beta[D]*x q[D]

"""

import math

import numpy as np
#np.set_printoptions(precision=6, linewidth=300)

from ....toolbox import toolbox
from ..dataset import Dataset
from .forward import ForwardModel


class LWLRForwardModel(ForwardModel):
    """Locally Weighted Linear Regression Forward Model"""

    name = 'LWLR'
    desc = 'LWLR, Locally Weighted Linear Regression'

    def __init__(self, dim_x, dim_y, sigma = 1.0, k = None, **kwargs):
        """Create the forward model

        @param dim_x    the input dimension
        @param dim_y    the output dimension
        @param sigma    sigma for the guassian distance.
        @param nn       the number of nearest neighbors to consider for regression.
        """
        self.k        = k or 2*dim_x+1
        ForwardModel.__init__(self, dim_x, dim_y, sigma = sigma, k = self.k, **kwargs)
        self.sigma_sq = sigma*sigma

    @property
    def sigma(self):
        return self.conf['sigma']

    @sigma.setter
    def sigma(self, sigma):
        self.sigma_sq = sigma*sigma
        self.conf['sigma'] = sigma

    ### LWR regression


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
        w = self._weights(dists, sigma_sq)

        Xq  = np.array(np.append([1.0], xq), ndmin = 2)
        X   = np.array([self.dataset.get_x_padded(i) for i in index])
        Y    = np.array([self.dataset.get_y(i) for i in index])

        W   = np.diag(w)
        WX  = np.dot(W, X)
        WXT = WX.T

        B   = np.dot(np.linalg.pinv(np.dot(WXT, WX)),WXT)

        self.mat = np.dot(B, np.dot(W, Y))
        Yq  = np.dot(Xq, self.mat)

        return Yq.ravel()

    def _weights(self, dists, sigma_sq):

        w = np.fromiter((toolbox.gaussian_kernel(d/self.dim_x, sigma_sq)
                         for d in dists), np.float, len(dists))

        wsum = w.sum()
        if wsum == 0:
            return 1.0/len(dists)*np.ones((len(dists),))
        else:
            eps = wsum * 1e-10 / self.dim_x
            return np.fromiter((w_i/wsum if w_i > eps else 0.0 for w_i in w), np.float)

class ESLWLRForwardModel(LWLRForwardModel):
    """ES-LWLR : LWLR with estimated sigma, on a query basis, as the mean distance."""

    name = 'ES-LWLR'

    def _weights(self, dists, sigma_sq):
        sigma_sq=(dists**2).sum()/len(dists)
        return LWLRForwardModel._weights(self, dists, sigma_sq)

pLWLRForwardModel = None
pESLWLRForwardModel = None

def enable_cmodels():
    global LWLRForwardModel, pLWLRForwardModel
    global ESLWLRForwardModel, pESLWLRForwardModel
    try:
        from cdataset import cLWLRForwardModel
        pLWLRForwardModel = LWLRForwardModel # keeping the python version accessible
        LWLRForwardModel = cLWLRForwardModel

        class cESLWLRForwardModel(cLWLRForwardModel):
            """ES-LWLR : LWLR with estimated sigma, on a query basis, as the mean distance.
            Based on c implementation.
            """
            def __init__(self, *args, **kwargs):
                cLWLRForwardModel.__init__(self, *args, **kwargs)
                self.es = True

        pESLWLRForwardModel = ESLWLRForwardModel # keeping the python version accessible
        ESLWLRForwardModel = cESLWLRForwardModel

    except ImportError:
        # print("warning: cdataset.cLWLRForwardModel import error, defaulting to (slower) python implementation.")
        pass
        
def disable_cmodels():
    global LWLRForwardModel, ESLWLRForwardModel
    LWLRForwardModel = pLWLRForwardModel
    ESLWLRForwardModel = pESLWLRForwardModel
    print("cmodels disabled")

enable_cmodels()
