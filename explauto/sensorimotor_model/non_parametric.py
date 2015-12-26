
import numpy as np

from numpy import array

from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from .learner import Learner
from explauto.utils import bounds_min_max


class NonParametric(SensorimotorModel):
    """ This class wraps the non-parametric forward and inverse models implemented by Fabien Benureau, in order to fit into the Explauto framework. 
        Original code available at https://github.com/humm/models
        Adapted by Sebastien Forestier at https://github.com/sebastien-forestier/models
    """
    def __init__(self, conf, sigma_explo_ratio=0.1, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):

        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds', 'm_mins', 'm_maxs']:
            setattr(self, attr, getattr(conf, attr))

        self.sigma_expl = (conf.m_maxs - conf.m_mins) * float(sigma_explo_ratio)
        self.mode = 'explore'
        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))

        self.model = Learner(mfeats, sfeats, mbounds, fwd, inv, **learner_kwargs)
        self.t = 0

    def infer(self, in_dims, out_dims, x):
        if self.t < max(self.model.imodel.fmodel.k, self.model.imodel.k):
            raise ExplautoBootstrapError
        
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return array(self.model.predict_effect(tuple(x)))
        
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if self.mode == 'explore':
                self.mean_explore = array(self.model.infer_order(tuple(x)))
                r = np.random.normal(self.mean_explore, self.sigma_expl)
                res = bounds_min_max(r, self.m_mins, self.m_maxs)
                return res
            else:  # exploit'
                return array(self.model.infer_order(tuple(x)))                
            
        elif out_dims == self.m_dims[len(self.m_dims)/2:]:  # dm = i(M, S, dS)
            assert len(x) == len(in_dims)
            m = x[:self.m_ndims/2]
            s = x[self.m_ndims/2:][:self.s_ndims/2]
            ds = x[self.m_ndims/2:][self.s_ndims/2:]
            self.mean_explore = array(self.model.imodel.infer_dm(m, s, ds))               
            if self.mode == 'explore': 
                r = np.random.normal(self.mean_explore, self.sigma_expl[out_dims])
                res = bounds_min_max(r, self.m_mins[out_dims], self.m_maxs[out_dims])                
                return res       
            else:
                return self.mean_explore
        else:
            raise NotImplementedError
                
                

    def predict_given_context(self, x, c, c_dims):
        return self.model.imodel.fmodel.predict_given_context(x, c, c_dims)

    def update(self, m, s):
        self.model.add_xy(tuple(m), tuple(s))
        self.t += 1

    def update_batch(self, m_list, s_list):
        self.model.add_xy_batch(m_list, s_list)
        self.t += len(m_list)
        
    def size(self):
        return self.t


sensorimotor_models = {
    'NN': (NonParametric, {'default': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.1},
                           'exact': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':0.}}),
    'WNN': (NonParametric, {'default': {'fwd': 'WNN', 'inv': 'WNN', 'k':20, 'sigma':0.1}}),
    'LWLR-BFGS': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'inv': 'L-BFGS-B', 'maxfun':50}}),
    'LWLR-CMAES': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'inv': 'CMAES', 'cmaes_sigma':0.05, 'maxfevals':20}}),
}
