from numpy import array

from .npendulum import NPendulumEnvironment


def make_npendulum_config(nb_joints, m_ndims, m_max, s_mins, s_maxs, noise):
    return dict(n=nb_joints,
                m_mins=array([-m_max] * m_ndims),
                m_maxs=array([m_max] * m_ndims),
                s_mins=s_mins,
                s_maxs=s_maxs,
                noise=noise)

config1 = make_npendulum_config(5, 10, 0.1, array([-450., -1.]), array([450., 1.]), 0.02)

environment = NPendulumEnvironment
configurations = {'default': config1}


def testcases(config_str, n_samples=-1):
    return None
