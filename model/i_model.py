import numpy

import imle as imle_
import utils

class InterestModel(object):
    def __init__(self, bounds):
        self.bounds = bounds
        self.ndims = bounds.shape[1]

    def sample(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

class RandomInterest(InterestModel):
    def sample(self):
        return utils.rand_bounds(self.bounds)

    def update(self, x, err):
        pass

class ProgressInterest(InterestModel):
    def __init__(self, bounds, sigma0, psi0):
        InterestModel.__init__(self, bounds)

        self.imle = imle_.Imle(in_ndims=self.ndims+1, out_ndims=1,
                               sigma0=0.2**2., Psi0=psi0, p0=0.3, multiValuedSignificance=0.5)
        self.scale_t = 2. * 0.2/numpy.sqrt(sigma0)
        self.t = 0.
        self.gmm = imle_.GMM(n_components=1, covariance_type='full')
        self.gmm.weights = [1.0]
        self.gmm.means_ = numpy.zeros((1, self.ndims+2))
#TO FIX: covars_ in fct of sigma0 and Psi0 :
        variances = numpy.hstack((1. , (1./12.) * (0.1 * (bounds[1,:] - bounds[0,:])) ** 2, psi0))
        self.gmm.covars_ = numpy.array([numpy.diag(variances)])

    def sample(self):
        x = self.gmm_interest().sample()
        x = numpy.maximum(x, self.bounds[0, :])
        x = numpy.minimum(x, self.bounds[1, :])
        return x.T

    def update(self, x, comp):
        self.imle.update(numpy.hstack(([self.t], x.flatten())), [comp])
        self.t += self.scale_t
        self.update_gmm()
        return self.t, x, comp

    def update_gmm(self):
        self.gmm = self.imle.to_gmm()

    def gmm_interest(self):
        cov_t_c = numpy.array([self.gmm.covars_[k,0,-1] for k in range(self.gmm.n_components)])
        cov_t_c = numpy.exp(cov_t_c)
        #cov_t_c[cov_t_c <= 1e-100] = 1e-100
        gmm_choice = self.gmm.inference([0], range(1, self.ndims + 1), [self.t])
        gmm_choice.weights_ = cov_t_c
        gmm_choice.weights_ /= numpy.array(gmm_choice.weights_).sum()
        #gmm_choice = gmm_choice.inference([], [0], []) 
        return gmm_choice


