from numpy import pi, array
from numpy import random
from copy import copy

import simple_lip

from .. import Environment
from ...utils.utils import bounds_min_max
from ...models.motor_primitive import BasisFunctions


class NPendulumEnvironment(Environment):
    """This classe implements the n-pendulum environnement.
    """

    def __init__(self,  m_mins, m_maxs, s_mins, s_maxs, noise, n = 4):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.noise = noise
        self.n = n
        self.x0 = hstack(( 0,- pi / 2 * ones(n), 1e-3 * ones(n+1)))
        self.dt = 0.025
        self.bf = BasisFunctions(self.conf.m_ndims, 70*self.dt, self.dt, 4.)

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):
        """
        :note the duration of the movement is  70*self.dt
        :note to use basis functions rather than step functions, set :useBF: to 1
        """
        useBF = 0
        if useBF = 0:
            func = simulation.step(m, 70*self.dt)
        else:
            traj = self.bf.trajectory(m).flatten()
            func = lambda t: traj[int(70*self.dt * t)]
        s = copy(self.x0)
        s = simulation.simulate(n, x0, dt, func):
        res = array(s)
        res += self.noise * random.randn(*res.shape)
        return res