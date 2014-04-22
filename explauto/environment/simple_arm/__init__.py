from numpy import array, pi

from .simple_arm import SimpleArmEnvironment

m_ndims = 7
s_ndims = 2  # necessarily, this only implements an arm moving in a 2D plan
m_mins = array([-pi/3.] * m_ndims)
s_mins = array([-1.] * s_ndims)

test_config = dict(m_mins=m_mins,
                   m_maxs=-1.*m_mins,
                   s_mins=s_mins,
                   s_maxs=-1.*s_mins,
                   length_ratio=3.,
                   noise=0.02
                   )

environment = SimpleArmEnvironment
configurations = {'default': test_config}
