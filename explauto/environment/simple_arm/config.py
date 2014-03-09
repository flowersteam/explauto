from .. import Environment
from numpy import pi, array, ones
from ...utils.utils import bounds_min_max
import simple_arm 
from numpy.random import randn

def lengths(n_dofs, ratio):
    l=ones(n_dofs)
    for i in range(1,n_dofs - 1):
        l[i]=l[i-1]/ratio
    return l/sum(l)


m_ndims = 7
s_ndims = 2 #necessarily, this only implements an arm moving in a 2D plan
m_mins = array([-pi/3.] * m_ndims)
s_mins = array([-1.] * s_ndims)

test_config = dict(
                m_ndims = m_ndims,
                s_ndims = s_ndims,
                m_mins = m_mins,
                m_maxs = -1. * m_mins,
                s_mins = s_mins,
                s_maxs = -1. * s_mins,
                length_ratio = 3., 
                noise = 0.02
                )

class SimpleArmEnvironment(Environment):
    #def __init__(self, m_ndims, s_ndims, m_mins, m_maxs, length_ratio, noise):
    def __init__(self, **kwargs):
        for attr in ['m_mins', 'm_maxs', 's_mins', 's_maxs', 'length_ratio', 'noise']:
            setattr(self, attr, kwargs[attr])
        self.m_ndims =len(self.m_mins) 
        self.s_ndims =len(self.s_mins) 
        Environment.__init__(self, ndims=self.m_ndims + self.s_ndims)
        #self.m_mins = m_mins
        #self.m_maxs = m_maxs
        #self.noise = noise
        self.lengths = lengths(self.m_ndims, self.length_ratio)
        self.readable = range(self.m_ndims + self.s_ndims)
        self.writable =  range(self.m_ndims)

    def next_state(self, ag_state):
        m = ag_state
        self.state[:self.m_ndims] = bounds_min_max(m, self.m_mins, self.m_maxs)
        res = array(simple_arm.forward(self.state[:self.m_ndims], self.lengths))
        res += self.noise * randn(*res.shape)
        self.state[-s_ndims:] = res
        
