import importlib
import models
import numpy as np

class Robot(object):
    def __init__(self, exp):
        self.s_feats = tuple(range(exp.s_ndims))
        self.m_feats = tuple(range(-exp.m_ndims, 0))
        
        self.m_bounds = tuple([tuple(exp.ms_bounds[:, d].T) for d in exp.m_dims])
        self.env = exp.env
        
    def execute_order(self,order):
        return tuple(self.env.execute(order)[1].flatten())

    def test_set(self, n_tests):
        return models.testbed.testcase.uniform_sensor_testcases(self, n_tests)

