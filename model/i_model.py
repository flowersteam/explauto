from numpy.random import rand as rand
import numpy
#from gmminf import GMM

import imle as imle_

class RandomExploration:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.widths=self.bounds[1,:] - self.bounds[0,:]
    def sample(self, dims):
        return (self.widths[dims] * rand(1,len(dims)) + self.bounds[0,dims]).reshape(-1,1)
    def update(self, x, err):
        pass

class ProgressExploration:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.widths=self.bounds[1,:] - self.bounds[0,:]
        self.imle=imle_.Imle(**kwargs)
        self.t=0
        self.gmm=imle_.GMM(n_components=1, covariance_type='full')
        self.gmm.weights=[1.0]
        self.gmm.means_=numpy.zeros((1, self.in_ndims+1))
        self.gmm.covars_=numpy.array([numpy.diag(numpy.ones(self.in_ndims+1))]) 
    def sample(self, dims):
        self.gmm.weights=[self.gmm.covars_[k,0,-1] for k in range(self.gmm.n_components)]
        self.gmm.weights/=numpy.array(self.gmm.weights).sum()
        gmm_interest=self.gmm.inference([0],range(1,len(dims)+1), [(self.t+1)/self.scale_t])
        return gmm_interest.sample().T
    def update(self, x, comp):
        self.imle.update(numpy.hstack(([self.t/self.scale_t],x.flatten())), [comp])
        self.t+=1
        self.update_gmm()
        return self.t, x, comp
    def update_gmm(self):
        self.gmm=self.imle.to_gmm()
