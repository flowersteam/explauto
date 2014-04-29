from numpy import array, pi, hstack
from copy import copy

from .simple_arm import SimpleArmEnvironment

m_ndims = 7
s_ndims = 2  # necessarily, this only implements an arm moving in a 2D plan
m_mins = array([-pi/3.] * m_ndims)
s_mins = array([-0.5, -1.])
s_maxs = array([1., 1.])

test_config = dict(m_mins=m_mins,
                   m_maxs=-1.*m_mins,
                   s_mins=s_mins,
                   s_maxs=s_maxs,
                   length_ratio=3.,
                   noise=0.02
                   )

hd_config = copy(test_config)
hd_config['m_mins'] = array([-pi/12.] * 20)
hd_config['m_maxs'] = -1. * hd_config['m_mins']
hd_config['s_mins'] = array([0.2, -0.7])
hd_config['s_maxs'] = array([1., 0.7])

environment = SimpleArmEnvironment
configurations = {'default': test_config, 'high_dimensional': hd_config}
