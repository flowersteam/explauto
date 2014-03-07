from .. import Environment
from numpy import pi, array, ones
from ...models.utils import bounds_min_max
import simple_arm 
from numpy.random import randn

def lengths(n_dofs, ratio):
    l=ones(n_dofs)
    for i in range(1,n_dofs - 1):
        l[i]=l[i-1]/ratio
    return l/sum(l)

m_ndims = 7
s_ndims = 2 #necessarily, this only implements a arm moving in a 2D plan

m_mins = array([-pi/3.] * m_ndims)
m_maxs = -1. * m_mins

noise = 0.02

class SimpleArmEnvironment(Environment):
    def __init__(self, m_ndims = m_ndims, length_ratio = 3., m_mins = m_mins, m_maxs = -1. * m_mins, noise = noise):
        Environment.__init__(self, ndims=m_ndims + s_ndims)
        self.m_ndims = m_ndims
        self.m_mins = m_mins
        self.m_maxs = m_maxs
        self.noise = noise
        self.lengths = lengths(self.m_ndims, length_ratio)

    def next_state(self, ag_state):
        m = ag_state
        self.state[:self.m_ndims] = bounds_min_max(m, self.m_mins, self.m_maxs)
        res = array(simple_arm.forward(self.state[:self.m_ndims], self.lengths))
        res += self.noise * randn(*res.shape)
        self.state[-s_ndims:] = res
        
