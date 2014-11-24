# import numpy as np
from numpy import argmax, array

from .sensorimotor_model import SensorimotorModel
from ..models import imle_model as imle_
from ..models.gmminf import GMM
from ..exceptions import ExplautoBootstrapError

class Normalizer(object):
    def __init__(self, conf):
        self.conf = conf
    def normalize(self, data, dims):
        return (data - self.conf.mins[dims]) / self.conf.ranges[dims]

    def denormalize(self, data, dims):
        return (data * self.conf.ranges[dims]) + self.conf.mins[dims]

class ImleModel(SensorimotorModel):
    """
        This class wraps the IMLE model from Bruno Damas ( http://users.isr.ist.utl.pt/~bdamas/IMLE ) into a sensorimotor model class to be used by ..agent.agent
        """
    # def __init__(self, m_dims, s_dims, sigma0, psi0, mode='explore'):
    def __init__(self, conf, mode='explore', **kwargs_imle):
        """ :param list m_dims: indices of motor dimensions
            :param list_ndims: indices of sensory dimensions
            :param float sigma0: a priori variance of the linear models on motor dimensions
            :param list psi0: a priori variance of the gaussian noise on each sensory dimensions
            :param string mode: either 'exploit' or 'explore' (default 'explore') to choose if the infer(.) method will return the most likely output or will sample according to the output probability.
            .. note::
            """
        SensorimotorModel.__init__(self, conf)
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        if 'sigma0' not in kwargs_imle:  # sigma0 is None:
            kwargs_imle['sigma0'] = 1./30. # (conf.m_maxs[0] - conf.m_mins[0]) / 30.
        if 'Psi0' not in kwargs_imle:  # if psi0 is None:
            kwargs_imle['Psi0'] = array([1./30.] * conf.s_ndims) ** 2 # ((conf.s_maxs - conf.s_mins) / 30.)**2
        self.mode = mode
        self.t = 0
        self.imle = imle_.Imle(d=len(self.m_dims), D=len(self.s_dims), **kwargs_imle)
                               #sigma0=sigma0, Psi0=psi0)
        self.normalizer = Normalizer(conf)

    def infer(self, in_dims, out_dims, x_):
        x = self.normalizer.normalize(x_, in_dims)
        if self.t < 1:
            raise ExplautoBootstrapError
        if in_dims == self.s_dims and out_dims == self.m_dims:
            # try:
            res = self.imle.predict_inverse(x, var=True, weight=True)
            sols = res['prediction']
            covars = res['var']
            weights = res['weight']
            # sols, covars, weights = self.imle.predict_inverse(x)
            if self.mode == 'explore':
                gmm = GMM(n_components=len(sols), covariance_type='full')
                gmm.weights_ = weights / weights.sum()
                gmm.covars_ = covars
                gmm.means_ = sols
                return self.normalizer.denormalize(gmm.sample().flatten(), out_dims)
            elif self.mode == 'exploit':
                # pred, _, _, jacob = self.imle.predict(sols[0])
                sol = sols[argmax(weights)]  # .reshape(-1,1) + np.linalg.pinv(jacob[0]).dot(x - pred.reshape(-1,1))
                return self.normalizer.denormalize(sol, out_dims)

            # except Exception as e:
            #     print e
            #     return self.imle.to_gmm().inference(in_dims, out_dims, x).sample().flatten()

        # elif in_dims == self.m_dims and out_dims==self.s_dims:
        #     return self.imle.predict(x.flatten()).reshape(-1,1)
        else:
            return self.normalizer.denormalize(self.imle.to_gmm().inference(in_dims, out_dims, x).sample().flatten(), out_dims)

    def update(self, m_, s_):
        m = self.normalizer.normalize(m_, self.conf.m_dims)
        s = self.normalizer.normalize(s_, self.conf.s_dims)
        self.imle.update(m, s)
        self.t += 1


class ImleGmmModel(ImleModel):
    def update_gmm(self):
        self.gmm = self.imle.to_gmm()

    def infer(self, in_dims, out_dims, x):
        self.update_gmm()
        return self.gmm.inference(in_dims, out_dims, x).sample().T

def make_priors(prior_coef):
    priors = {}
    for prior in ['wsigma', 'wSigma', 'wNu', 'wLambda', 'wPsi']:
        priors[prior] = prior_coef
    return priors

configurations = {'default': {}, 'low_prior': make_priors(1.), 'hd_prior': make_priors(10.)}

sensorimotor_models = {'imle': (ImleModel, configurations)}
