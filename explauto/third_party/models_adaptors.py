""" Adaptors to Fabien Benureau's models library. """

from numpy import array


def configuration(env_or_ag):
    s_feats = tuple(range(env_or_ag.conf.s_ndims))
    m_feats = tuple(range(-env_or_ag.conf.m_ndims, 0))
    m_bounds = tuple([(env_or_ag.conf.m_mins[d], env_or_ag.conf.m_maxs[d]) for d in env_or_ag.conf.m_dims])
    return m_feats, s_feats, m_bounds


class Robot(object):
    def __init__(self, env, log=True):
        self.m_feats, self.s_feats, self.m_bounds = configuration(env)
        self.env = env
        self.log = log

    def execute_order(self, order):
        self.env.update(order, self.log)
        return tuple(self.env.state[self.env.conf.s_dims])


class Learner(object):
    def __init__(self, ag):
        self.m_feats, self.s_feats, self.m_bounds = configuration(ag)
        self.ag = ag

    def add_xy(self, x, y):
        self.ag.sensorimotor_model.update(x, y)

    def infer_order(self, goal, **kwargs):
        return self.ag.infer(self.ag.conf.s_dims, self.ag.conf.m_dims, array(goal))

    def predict_effect(self, order, **kwargs):
        return self.ag.infer(self.ag.conf.m_dims, self.ag.conf.s_dims, array(order))
