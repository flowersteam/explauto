# import numpy as np

from .sensorimotor_model import SensorimotorModel
from ..models import imle_model as imle_
from ..models.gmminf import GMM
from .. import ExplautoBootstrapError



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
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        if 'sigma0' not in kwargs_imle:  # sigma0 is None:
            kwargs_imle['sigma0'] = (conf.m_maxs[0] - conf.m_mins[0]) / 30.
        if 'Psi0' not in kwargs_imle:  # if psi0 is None:
            kwargs_imle['Psi0'] = ((conf.s_maxs - conf.s_mins) / 100.)**2
        if 'wPsi0' not in kwargs_imle:  # if psi0 is None:
            kwargs_imle['wPsi'] = 1.2 ** conf.m_ndims
            print kwargs_imle['wPsi']
        self.mode = mode
        self.t = 0
        self.imle = imle_.Imle(in_ndims=len(self.m_dims), out_ndims=len(self.s_dims), **kwargs_imle)
                               #sigma0=sigma0, Psi0=psi0)

    def infer(self, in_dims, out_dims, x):
        if self.t < 1:
            raise ExplautoBootstrapError
        if in_dims == self.s_dims and out_dims == self.m_dims:
            try:
                sols, covars, weights = self.imle.predict_inverse(x)
                if self.mode == 'explore':
                    gmm = GMM(n_components=len(sols), covariance_type='full')
                    gmm.weights_ = weights / weights.sum()
                    gmm.covars_ = covars
                    gmm.means_ = sols

                    return gmm.sample()
                elif self.mode == 'exploit':
                    # pred, _, _, jacob = self.imle.predict(sols[0])
                    sol = sols[0]  # .reshape(-1,1) + np.linalg.pinv(jacob[0]).dot(x - pred.reshape(-1,1))
                    return sol

            except Exception as e:
                print e
                return self.imle.to_gmm().inference(in_dims, out_dims, x).sample()

        # elif in_dims == self.m_dims and out_dims==self.s_dims:
        #     return self.imle.predict(x.flatten()).reshape(-1,1)
        else:
            return self.imle.to_gmm().inference(in_dims, out_dims, x).sample()

    def update(self, m, s):
        self.imle.update(m, s)
        self.t += 1


class ImleGmmModel(ImleModel):
    def update_gmm(self):
        self.gmm = self.imle.to_gmm()

    def infer(self, in_dims, out_dims, x):
        self.update_gmm()
        return self.gmm.inference(in_dims, out_dims, x).sample().T

low_prior_coef = 1
low_prior = {}
for prior in ['wsigma', 'wSigma', 'wNu', 'wLambda', 'wPsi']:
    low_prior[prior] = low_prior_coef

configurations = {'default': {}, 'low_prior': low_prior}
sensorimotor_models = {'imle': (ImleModel, configurations)}
