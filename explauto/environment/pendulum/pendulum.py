from numpy import pi, array
from numpy import random
from copy import copy

import simple_lip

from .. import Environment
from ...utils.utils import bounds_min_max
from ...models.motor_primitive import BasisFunctions


class PendulumEnvironment(Environment):
    def __init__(self,  m_mins, m_maxs, s_mins, s_maxs, noise):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.noise = noise
        self.x0 = [-pi, 0.]
        self.dt = 0.25
        self.bf = BasisFunctions(self.conf.m_ndims, 70*self.dt, self.dt, 4.)

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self):
        s = copy(self.x0)
        for u in self.bf.trajectory(self.state[:self.conf.m_ndims]).flatten():
            s = simple_lip.simulate(s, [u], self.dt)
        res = array(s)
        res += self.noise * random.randn(*res.shape)
        return res
