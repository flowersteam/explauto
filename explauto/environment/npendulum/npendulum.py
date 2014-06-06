from numpy import pi, array
from numpy import random, hstack, ones, zeros
from copy import copy

import simulation

from .. import Environment
from ...utils.utils import bounds_min_max
from ...models.motor_primitive import BasisFunctions


class NPendulumEnvironment(Environment):
    """This classe implements the n-pendulum environnement.
    """

    def __init__(self,  m_mins, m_maxs, s_mins, s_maxs, n, noise):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.noise = noise
        self.n = n
        self.x0 = hstack(( 0, -pi / 2 * ones(n), zeros(n+1)))
        self.dt = 0.01
        self.bf = BasisFunctions(self.conf.m_ndims, 70*self.dt, self.dt, 4.)

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):
        """
        :note the duration of the movement is  1000*self.dt
        :note to use basis functions rather than step functions, set :useBF: to 1
        """
        useBF = 0
        if useBF == 0:
            func = simulation.step(m, 1000*self.dt)
        else:
            traj = self.bf.trajectory(m).flatten()
            func = lambda t: traj[int(1000*self.dt * t)]
        s = simulation.simulate(self.n, self.x0, self.dt, func)
        s_cartesian = simulation.cartesian(self.n, s)
        res = array([s_cartesian[self.n], s_cartesian[2*self.n]])
        res += self.noise * random.randn(*res.shape)
        return res