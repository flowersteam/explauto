from numpy import array

from .sm_model import SmModel
from ..third_party.models.models.learner import Learner


class NonParametric(SmModel):
    """This class wraps the non-parametric forward and inverse models implemented by Fabien Benureau, in order to fit into the Explauto framework. Original code by Fabien available here: https://github.com/humm/models

    """
    def __init__(self, conf, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds']:
            setattr(self, attr, getattr(conf, attr))

        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))

        self.model = Learner(mfeats, sfeats, mbounds, fwd, inv, **learner_kwargs)

    def infer(self, in_dims, out_dims, x):
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return array(self.model.predict_effect(tuple(x.flatten())))
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            return array(self.model.infer_order(tuple(x.flatten())))
        else:
            raise NotImplementedError("NonParameticModel only implements forward (M -> S)"
                                      "and inverse (S -> M) model, not general prediction")

    def update(self, m, s):
        self.model.add_xy(tuple(m), tuple(s))


sensorimotor_model = NonParametric
configurations = {'default': {'fwd': 'LWLR', 'inv': 'L-BFGS-B'}}