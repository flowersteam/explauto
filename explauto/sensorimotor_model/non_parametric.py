
import numpy as np

from numpy import array

from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from .learner import Learner


class NonParametric(SensorimotorModel):
    """ This class wraps the non-parametric forward and inverse models implemented by Fabien Benureau, in order to fit into the Explauto framework. 
        Original code available at https://github.com/humm/models
        Adapted by Sebastien Forestier at https://github.com/sebastien-forestier/models
    """
    def __init__(self, conf, sigma_explo_ratio=0.05, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):

        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds']:
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
                return np.random.normal(self.mean_explore, self.sigma_expl)
            else:  # exploit'
                return array(self.model.infer_order(tuple(x)))                
            
        else:  # general prediction, no exploration noise
            assert len(x) == len(in_dims)
            dims_x = [d for d in in_dims if d < self.model.imodel.dim_x]
            dims_y = [d for d in in_dims if d >= self.model.imodel.dim_x]
            xi = array(x)[:len(dims_x)]
            yi = array(x)[len(dims_x):]
            self.mean_explore = array(self.model.imodel.infer_dims(xi, yi, dims_x, dims_y, out_dims))            
            return self.mean_explore

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
    'NN': (NonParametric, {'default': {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':1.0}}),
    'WNN': (NonParametric, {'default': {'fwd': 'WNN', 'inv': 'WNN', 'k':20, 'sigma':0.1}}),
    'LWLR-BFGS': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'inv': 'L-BFGS-B', 'maxfun':50}}),
    'LWLR-CMAES': (NonParametric, {'default': {'fwd': 'LWLR', 'k':10, 'inv': 'CMAES', 'cmaes_sigma':0.05, 'maxfevals':20}}),
}
