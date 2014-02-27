import numpy as np
import imle as imle_
from gmminf import GMM

class SmModel(object):
    def __init__(self, m_dims, s_dims):
        self.m_dims = m_dims
        self.s_dims = s_dims

    def infer(self, in_dims, out_dims):
        raise NotImplementedError

    def update(self, m, s):
        raise NotImplementedError


class ImleModel(SmModel):
    def __init__(self, m_dims, s_dims, sigma0, psi0, mode='explore'):
        SmModel.__init__(self, m_dims, s_dims)
        self.mode = mode
        self.imle = imle_.Imle(in_ndims=len(m_dims), out_ndims=len(s_dims),
                               sigma0=sigma0, Psi0=psi0)

    def infer(self, in_dims, out_dims,x):
        if in_dims == self.s_dims and out_dims == self.m_dims:
            try:
                sols, covars, weights = self.imle.predict_inverse(x.flatten())
                if self.mode == 'explore':
                    gmm = GMM(n_components=len(sols), covariance_type='full')
                    gmm.weights_ = weights / weights.sum()
                    gmm.covars_ = covars
                    gmm.means_ = sols

                    return gmm.sample().reshape(-1,1)
                elif self.mode == 'exploit':
                    #pred, _, _, jacob = self.imle.predict(sols[0])
                    sol = sols[0]#.reshape(-1,1) + np.linalg.pinv(jacob[0]).dot(x - pred.reshape(-1,1))
                    return sol

            except Exception as e:
                print e
                return self.imle.to_gmm().inference(in_dims, out_dims, x).sample().T

        elif in_dims == self.m_dims and out_dims==self.s_dims:
            return self.imle.predict(x.flatten()).reshape(-1,1)
        else:
            return self.imle.to_gmm().inference(in_dims, out_dims, x).sample().T

    def update(self, m, s):
        self.imle.update(m.flatten(), s.flatten())

class ImleGmmModel(ImleModel):
    def update_gmm(self):
        self.gmm = self.imle.to_gmm()

    def infer(self, in_dims,out_dims,x):
        self.update_gmm()
        return self.gmm.inference(in_dims, out_dims, x).sample().T

