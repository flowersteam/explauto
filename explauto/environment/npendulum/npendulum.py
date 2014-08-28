import shutil
import os

from numpy import pi, array, random, hstack, ones, zeros
from matplotlib.pyplot import savefig, clf, axes

import simulation

from ..environment import Environment
from ...utils.utils import bounds_min_max
from ...models.motor_primitive import BasisFunctions


class NPendulumEnvironment(Environment):
    """ This class implements the n-pendulum environnement.

    For more information, please look at :func:`~explauto.environment.npendulum.simulation.simulate`.

    """
    def __init__(self, n, m_mins, m_maxs, s_mins, s_maxs, noise):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.noise = noise
        self.n = n
        self.x0 = hstack((0, -pi / 2 * ones(n), zeros(n + 1)))
        self.dt = 0.01
        self.bf = BasisFunctions(self.conf.m_ndims, 1000*self.dt, self.dt, 4.)
        self.use_basis_functions = False

    def compute_motor_command(self, ag_state):
        return bounds_min_max(ag_state, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):
        """ This function generates the end effector position at the end of the movement.

        .. note:: The duration of the movement is  1000*self.dt
        .. note:: To use basis functions rather than step functions, set :use_basis_functions: to 1
        """
        if self.use_basis_functions:
            func = simulation.step(m, 1000*self.dt)
        else:
            traj = self.bf.trajectory(m).flatten()
            func = lambda t: traj[int(1000*self.dt * t)]

        states = simulation.simulate(self.n, self.x0, self.dt, func)
        last_state_cartesian = simulation.cartesian(self.n, states[-1])

        end_effector_pos = array([last_state_cartesian[self.n], last_state_cartesian[2*self.n]])
        end_effector_pos += self.noise * random.randn(len(end_effector_pos))
        return end_effector_pos

    def plot_s(self, ax, s):
        ax.plot(s[0], s[1], 's', ms=6)
        ax.axis([self.conf.s_mins[0], self.conf.s_maxs[0], self.conf.s_mins[1], self.conf.s_maxs[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def plot_and_compute(self, ax, m):
        s = self.compute_sensori_effect(m)
        ax.plot(s[0], s[1], 's', ms=6)
        ax.axis([self.conf.s_mins[0], self.conf.s_maxs[0],
                 self.conf.s_mins[1], self.conf.s_maxs[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def plot_npendulum(self, ax, m, time=999):
        func = simulation.step(m, 1000*self.dt)
        s = simulation.simulate(self.n, self.x0, self.dt, func)
        s_cartesian = simulation.cartesian(self.n, s[time])
        x = s_cartesian[:len(s_cartesian) / 2]
        y = s_cartesian[len(s_cartesian) / 2:]
        ax.plot(x, y)
        ax.plot(x[0], y[0], 'o', ms=6)
        ax.plot(x[-1], y[-1], 's', ms=6)

    def animate_pendulum(self, m,  path="anim_npendulum"):
        """This function generates few images at different instants in order to animate the pendulum.

        ..note:: this function destroys and creates the folder :anim_npendulum:.
        """

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        for t in range(0, 1000, 32):
            ax = axes()
            self.plot_npendulum(ax, m, t)
            savefig(os.path.join(path, "{}.png".format(t)))
            clf()
