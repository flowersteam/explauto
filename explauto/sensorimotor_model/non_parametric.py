import numpy as np

from numpy import array
from sklearn.mixture import sample_gaussian
from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from models.learner import Learner


class NonParametric(SensorimotorModel):
    """This class wraps the non-parametric forward and inverse models implemented by Fabien Benureau, in order to fit into the Explauto framework. Original code available here: https://github.com/humm/models

    """
    def __init__(self, conf, sigma_ratio=0.05, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):

        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds']:
            setattr(self, attr, getattr(conf, attr))

        self.sigma_expl = (conf.m_maxs - conf.m_mins) * float(sigma_ratio)
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
            return array(self.model.predict_effect(tuple(x.flatten())))
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if self.mode == 'explore':
#                 print "LWLR"
#                 print "sg", x
#                 dists, idxs = self.model.imodel.fmodel.dataset.nn_y(x)
#                 snn=array(self.model.imodel.fmodel.dataset.get_y(idxs[0]))
#                 print "snn", snn
#                 print "x guess", array(self.model.imodel.fmodel.dataset.get_x(idxs[0]))
                self.mean_explore = array(self.model.infer_order(tuple(x.flatten())))
#                 print "m", self.mean_explore
                res = sample_gaussian(self.mean_explore, self.sigma_expl ** 2)
#                 print "m + eps", res
#                 print "p(m)", array(self.model.predict_effect(tuple(self.mean_explore.flatten())))
#                 d1 = np.linalg.norm(array(self.model.predict_effect(tuple(self.mean_explore.flatten()))) - x)
#                 d2 = np.linalg.norm(snn - x)
#                 print "dist(p(m), sg) =", d1
#                 print "dist(snn, sg) =", d2
#                 if d1 > d2:
#                     print "-------- OPTIM FAILED"
#                 else:
#                     print "-------- OPTIM SUCCEED"
#                 print "sp", array(self.model.predict_effect(tuple(res.flatten())))
                #print "sp, snn", array(self.model.predict_effect(tuple(res.flatten()))), array(self.model.predict_effect(tuple(self.mean_explore.flatten())))
                return res, array(self.model.predict_effect(tuple(res.flatten()))), array(self.model.predict_effect(tuple(self.mean_explore.flatten()))) # m, sp, snn
        elif in_dims in self.s_dims and out_dims == self.m_dims:  # partial inverse, exploit
            res = array(self.model.infer_order(tuple(x.flatten()))) # TOCHECK
            sp = array(self.model.predict_effect(tuple(res.flatten())))
            return res, sp, sp
        else:
            raise NotImplementedError("NonParametic only implements forward (M -> S)"
                                      "and inverse (S -> M) model, not general prediction")

    def update(self, m, s):
        self.model.add_xy(tuple(m), tuple(s))
        self.t += 1

    def update_batch(self, m_list, s_list):
        self.model.add_xy_batch(m_list, s_list)
        self.t += len(m_list)
    

configurations = {'LWLR-BFGS': {'fwd': 'LWLR', 'inv': 'L-BFGS-B'},
                  'ES-LWLR-BFGS': {'fwd': 'ES-LWLR', 'inv': 'L-BFGS-B'},
                  'ES-WNN': {'fwd': 'ES-WNN', 'inv': 'ES-WNN'},
                  'WNN': {'fwd': 'WNN', 'inv': 'WNN'},
                  'AvgNN': {'fwd': 'AvgNN', 'inv': 'AvgNN'},
                  # 'NN': {'fwd': 'NN', 'inv': 'NN'},
                  'ES-WNN-BFGS': {'fwd': 'ES-WNN', 'inv': 'L-BFGS-B'}
                  }
configurations['default'] = configurations['LWLR-BFGS']

sensorimotor_models = {
    'non_parametric': (NonParametric, configurations),
    'knn': (NonParametric, {'default': {'fwd': 'ES-WNN', 'inv': 'ES-WNN'}}),
}
