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

class Explauto(object):
    def __init__(self, experiment):
        self.exp = importlib.import_module(experiment)
        self.robot = Robot(self.exp)
        self.hist_len = len(self.exp.i_dims) + len(self.exp.inf_dims) + self.exp.m_ndims + self.exp.s_ndims + 1

    def run(self):
        in_env = self.ag.produce()
        out_env = self.env.execute(in_env)
        return np.vstack(self.exp.ag.explore(self.exp.i_dims,self.exp.inf_dims)).reshape(-1) 
        #return np.array([np.vstack(self.exp.ag.explore(self.exp.i_dims,self.exp.inf_dims)).T for _ in range(n_runs)])
        
