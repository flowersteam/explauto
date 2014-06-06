from numpy import array

from ...utils import rand_bounds
from .npendulum import NPendulumEnvironment


def make_npendulum_config(m_ndims, m_max, s_mins, s_maxs, nb_joints, noise):
    return dict(m_mins=array([-m_max] * m_ndims),
                m_maxs=array([m_max] * m_ndims),
                s_mins=s_mins,
                s_maxs=s_maxs,
                n=nb_joints,
                noise=noise)

config1 = make_npendulum_config(10, 0.1, array([-15., -1.]), array([15., 1.]), 5, 0.02)

environment = NPendulumEnvironment
configurations = {'default': config1}

def testcases(config_str, n_samples=-1):
    return None
