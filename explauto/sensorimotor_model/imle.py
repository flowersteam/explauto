# import numpy as np

from .sensorimotor_model import SensorimotorModel
from ..models import imle_model as imle_
from ..models.gmminf import GMM


class ImleModel(SensorimotorModel):
    """
        This class wraps the IMLE model from Bruno Damas ( http://users.isr.ist.utl.pt/~bdamas/IMLE ) into a sensorimotor model class to be used by ..agent.agent
        """
    # def __init__(self, m_dims, s_dims, sigma0, psi0, mode='explore'):
    def __init__(self, conf, sigma0=None, psi0=None, mode='explore'):
        """ :param list m_dims: indices of motor dimensions
            :param list_ndims: indices of sensory dimensions
            :param float sigma0: a priori variance of the linear models on motor dimensions
            :param list psi0: a priori variance of the gaussian noise on each sensory dimensions
            :param string mode: either 'exploit' or 'explore' (default 'explore') to choose if the infer(.) method will return the most likely output or will sample according to the output probability.
            .. note::
            """
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        if sigma0 is None:
            sigma0 = (conf.m_maxs[0] - conf.m_mins[0]) / 30.
        if psi0 is None:
            psi0 = ((conf.s_maxs - conf.s_mins) / 30.)**2
        self.mode = mode
        self.imle = imle_.Imle(in_ndims=len(self.m_dims), out_ndims=len(self.s_dims),
                               sigma0=sigma0, Psi0=psi0)

    def infer(self, in_dims, out_dims, x):
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


class ImleGmmModel(ImleModel):
    def update_gmm(self):
        self.gmm = self.imle.to_gmm()

    def infer(self, in_dims, out_dims, x):
        self.update_gmm()
        return self.gmm.inference(in_dims, out_dims, x).sample().T

sensorimotor_model = ImleModel
configurations = {'default': {}}
