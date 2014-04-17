import importlib
import models
from ..utils.config import Configuration 
import numpy as np

class Robot(object):
    def __init__(self, env):
        #self.conf = Configuration(env.params)
        self.s_feats = tuple(range(env.s_ndims))
        self.m_feats = tuple(range(-env.m_ndims, 0))
        
        self.m_bounds = tuple([tuple(env.bounds[:, d].T) for d in env.m_dims])
        self.env = env
        
    def execute_order(self,order):
        self.env.next_state(order)
        return tuple(self.env.state[self.env.s_dims])

    def test_set(self, n_tests):
        return models.testbed.testcase.uniform_sensor_testcases(self, n_tests)

