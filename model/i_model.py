import numpy

import imle as imle_
import utils

class InterestModel(object):
    def __init__(self, bounds):
        self.bounds = bounds

    def sample(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

class RandomInterest(InterestModel):
    def sample(self, dims):
        return utils.rand_bounds(self.bounds)

    def update(self, x, err):
        pass

class ProgressInterest(InterestModel):
    def __init__(self, bounds, sigma0, psi0):
        InterestModel.__init__(self, bounds)

        self.imle = imle_.Imle(sigma0, psi0)
        self.t = 0
        self.ndims = self.bounds.shape[1]
        self.gmm = imle_.GMM(n_components=1, covariance_type='full')
        self.gmm.weights = [1.0]
        self.gmm.means_ = numpy.zeros((1, self.ndims+1))
        self.gmm.covars_ = numpy.array([numpy.diag(numpy.ones(self.nndims + 1))])

    def sample(self):
        self.gmm.weights = [self.gmm.covars_[k,0,-1] for k in range(self.gmm.n_components)]
        self.gmm.weights /= numpy.array(self.gmm.weights).sum()
        gmm_interest = self.gmm.inference([0], range(1, self.ndims + 1), [(self.t + 1) / self.scale_t])
        return gmm_interest.sample().T

    def update(self, x, comp):
        self.imle.update(numpy.hstack(([self.t / self.scale_t], x.flatten())), [comp])
        self.t += 1
        self.update_gmm()
        return self.t, x, comp

    def update_gmm(self):
        self.gmm = self.imle.to_gmm()
