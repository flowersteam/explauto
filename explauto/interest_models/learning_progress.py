from interest_model import InterestModel
import numpy
from collections import deque
from ..models.gmminf import GMM
from copy import deepcopy
from ..utils import discrete_random_draw

class ProgressInterest(InterestModel):
    def __init__(self, i_dims, bounds, sigma0, psi0, measure):
        import imle as imle_
        InterestModel.__init__(self, i_dims)
        self.measure = measure

        self.imle = imle_.Imle(in_ndims=self.ndims+1, out_ndims=1,
                               sigma0=0.2**2., Psi0=psi0, p0=0.3, multiValuedSignificance=0.5)
        self.scale_t = 2. * 0.2/numpy.sqrt(sigma0)
        self.t = 0.
        self.gmm = GMM(n_components=1, covariance_type='full')
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

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        self.imle.update(numpy.hstack(([self.t], x.flatten())), [measure])
        self.t += self.scale_t
        self.update_gmm()
        return self.t, xy[self.i_dims], measure

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

class GmmInterest(InterestModel):
    def __init__(self, i_dims, bounds, sigma0, psi0, measure):
        InterestModel.__init__(self, i_dims)

        self.measure = measure
        self.scale_t = 2. * 0.2/numpy.sqrt(sigma0)
        self.t = 0.
        self.gmm = GMM(n_components=12, covariance_type='full')
        self.gmm.weights_ = [1.0] * self.gmm.n_components
        self.gmm.means_ = numpy.zeros((self.gmm.n_components, self.ndims+2))
#TO FIX: covars_ in fct of sigma0 and Psi0 :
        variances = numpy.hstack((1. , (1./12.) * (0.1 * (bounds[1,:] - bounds[0,:])) ** 2, psi0))
        self.gmm.covars_ = numpy.array([numpy.diag(variances)] * self.gmm.n_components)
        self.data = []
        self.nb_data = 1000
        self.n_t = 0

    def sample(self):
        x = self.gmm_interest().sample()
        x = numpy.maximum(x, self.bounds[0, :])
        x = numpy.minimum(x, self.bounds[1, :])
        return x.T

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        if len(self.data) > self.nb_data:
            self.data.remove(self.data[0])
        self.data.append(numpy.hstack(([self.t], x.flatten(), [measure])))
        self.t += self.scale_t
        if abs(self.t % (self.nb_data * self.scale_t / 4.)) < self.scale_t:
            print 'upd', self.t 
            self.update_gmm()
        return self.t, xy[self.i_dims], measure

    def update_gmm(self):
        self.gmm =  GMM(n_components=12, covariance_type='full')
        self.gmm.fit(numpy.array(self.data))

    def gmm_interest(self):
        cov_t_c = numpy.array([self.gmm.covars_[k,0,-1] for k in range(self.gmm.n_components)])
        cov_t_c = numpy.exp(cov_t_c)
        #cov_t_c[cov_t_c <= 1e-100] = 1e-100
        gmm_choice = self.gmm.inference([0], range(1, self.ndims + 1), [self.t])
        gmm_choice.weights_ = cov_t_c
        gmm_choice.weights_ /= numpy.array(gmm_choice.weights_).sum()
        #gmm_choice = gmm_choice.inference([], [0], []) 
        return gmm_choice

class DiscreteProgressInterest(InterestModel):
    def __init__(self, i_dims, x_card, win_size, measure):
        InterestModel.__init__(self, i_dims)
        self.measure = measure
        self.win_size = win_size
        #self.t = [win_size] * self.xcard
        queue =  deque([0. for t in range(win_size)], maxlen = win_size)
        self.queues = [deepcopy(queue) for _ in range(x_card)]
        #self.queues = [ deque([[t, numpy.random.rand()] for t in range(win_size)], maxlen = win_size) for _ in range(x_card)]
        #self.queues = [ deque([[t, 1.] for t in range(win_size)], maxlen = win_size) for _ in range(x_card)]
        self.choices = numpy.zeros((10000, len(i_dims)))
        self.comps = numpy.zeros(10000)
        self.t = 0

    def progress(self):
        return numpy.array([numpy.cov(zip(range(self.win_size), q), rowvar=0)[0,1] for q in self.queues])

    def sample(self):
        w =  abs(self.progress())
        w = numpy.exp(3. * w) / numpy.exp(3.)
        return discrete_random_draw(w)

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        self.queues[int(xy[self.i_dims])].append(measure)
        self.choices[self.t,:] = xy[self.i_dims]
        self.comps[self.t] = measure
        self.t += 1


class BayesOptInterest(InterestModel):
    def __init__(self, i_dims):
        InterestModel.__init__(self, i_dims)
        data = []

    def sample(self):
        raise NotImplementedError

    def update(self, x, arg_measure):
        raise NotImplementedError
