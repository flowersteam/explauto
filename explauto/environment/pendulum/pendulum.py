from numpy import pi, array, cos, sin
from numpy import random
from copy import copy

import simple_lip

from ..environment import Environment
from ...utils.utils import bounds_min_max
from ...models.motor_primitive import BasisFunctions


class PendulumEnvironment(Environment):
    def __init__(self,  m_mins, m_maxs, s_mins, s_maxs, noise, torque_max=0.25, dt = 0.25):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.noise = noise
        self.x0 = [-pi, 0.]
        self.dt = dt
        self.torque_max = torque_max
        self.bf = BasisFunctions(self.conf.m_ndims, 70*self.dt, self.dt, 4.)
        self.x = copy(self.x0)

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def reset(self):
        self.x = copy(self.x0)

    def apply_torque(self, u):
        self.x = simple_lip.simulate(self.x, [u], self.dt)

    def compute_sensori_effect(self, m):
        # self.x = copy(self.x0)
        for u in self.bf.trajectory(m).flatten():
            self.apply_torque(u)
        res = array(self.x)
        res += self.noise * random.randn(*res.shape)
        return res

    def plot_current_state(self, ax):
        ax.plot(0, 0, 'sk', ms=6)
        x, y = cos(self.x[0] + pi/2.), sin(self.x[0] + pi/2.)
        ax.plot(x, y, 'ok', ms=16)
        ax.plot([0, x], [0, y], lw=4, color='grey')
        ax.axis([-1.2, 1.2, -1.2, 1.2])
