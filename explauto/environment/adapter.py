""" Adaptors to Fabien Benureau's models library. """

import importlib
from ..utils.config import Configuration
import numpy as np

#from ..third_party.models.models import testbed

def configuration(env_or_ag):
    s_feats = tuple(range(env_or_ag.s_ndims))
    m_feats = tuple(range(-env_or_ag.m_ndims, 0))
    m_bounds = tuple([(env_or_ag.m_mins[d], env_or_ag.m_maxs[d]) for d in conf['m_dims']])
    return m_feats, s_feats, m_bounds

class Robot(object):
    def __init__(self, env):
        self.m_feats, self.s_feats, self.m_bounds = configuration(env)
        self.env = env

    def execute_order(self,order):
        self.env.next_state(order)
        return tuple(self.env.state[self.env.s_dims])

    #def test_set(self, n_tests):
        #return testbed.testcase.uniform_sensor_testcases(self, n_tests)

class Learner(object):
    def __init__(self, ag):
        self.m_feats, self.s_feats, self.m_bounds = configuration(ag)
        self.ag = ag


