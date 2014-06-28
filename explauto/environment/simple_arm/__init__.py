from numpy import array, pi, sqrt, cos, sin, linspace, zeros
from numpy import random


from ... import ExplautoNoTestCasesError
from .simple_arm import SimpleArmEnvironment


def make_arm_config(m_ndims, m_max, s_mins, s_maxs, length_ratio, noise):
    return dict(m_mins=array([-m_max] * m_ndims),
                m_maxs=array([m_max] * m_ndims),
                s_mins=s_mins,
                s_maxs=s_maxs,
                length_ratio=float(length_ratio),
                noise=noise)

low_dim = make_arm_config(3, pi/3, array([-0.5, -1.]), array([1., 1.]), 3, 0.02)
mid_dim = make_arm_config(7, pi/3, array([-0.5, -1.]), array([1., 1.]), 3, 0.02)
hd_dim = make_arm_config(50, pi/12., array([-1., -1.]), array([1., 1.]), 1., 0.02)
# hd_dim = make_arm_config(30, pi/8., array([-0.6, -0.9]), array([1., 0.9]), 1., 0.001)

# hd_dim_range = make_arm_config(30, pi/8., array([-2., -2.]), array([2., 2.]), 1., 0.001)
# hd_dim_range = make_arm_config(15, pi/3., array([-2., -2.]), array([2., 2.]), 2./3., 0.001)
hd_dim_range = make_arm_config(50, pi/12., array([-2., -2.]), array([2., 2.]), 1., 0.02)

environment = SimpleArmEnvironment
configurations = {'low_dimensional': low_dim,
                  'mid_dimensional': mid_dim,
                  'high_dimensional': hd_dim,
                  'high_dim_high_s_range': hd_dim_range,
                  'default': low_dim}


def testcases(config_str, n_samples=100):
    tests = zeros((n_samples, 2))
    #FIXME low_dimensional
    if config_str in ('high_dimensional', 'high_dim_high_s_range'):
        i = 0
        for r, theta in array([1., 2*pi]) * random.rand(n_samples, 2) + array([0., -pi]):
            tests[i, :] = sqrt(r) * array([cos(theta), sin(theta)])
            i += 1
        return tests

    else:
        env = environment(**configurations[config_str])
        env.noise = 0.
        return env.uniform_sensor(n_samples)
