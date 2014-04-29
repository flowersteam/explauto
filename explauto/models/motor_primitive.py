from numpy import linspace, array, arange, tile, dot, zeros

from .gaussian import Gaussian
from ..utils import rk4


class BasisFunctions(object):
    def __init__(self, n_basis, duration, dt, sigma):
        self.n_basis = n_basis

        means = linspace(0, duration, n_basis)
        variances = duration / (sigma * n_basis)**2
        gaussians = [Gaussian(array([means[k]]), array([[variances]]))
                     for k in range(len(means))]

        self.x = arange(0., duration, dt)
        y = array([gaussians[k].normal(self.x.reshape(-1, 1)) for k in range(len(means))])
        self.z = y / tile(sum(y, 0), (n_basis, 1))

    def trajectory(self, weights):
        return dot(weights, self.z)


class MovementPrimitive(object):
    def __init__(self, duration, n_basis, dt, stiffness=0., damping=0.):
        """
        :param float duration: duration of the movement in seconds
        :param list dt: time step used for numerical integration
        """
        self.dt = dt
        self.duration = duration
        self.stiffness = stiffness
        self.damping = damping
        self.basis = BasisFunctions(n_basis, self.duration, dt, 2.)
        self.traj = zeros((self.duration/dt, 3))
        self.acc = zeros(self.duration/dt)  # +1 due to ..utils.rk4 implementation

    def acceleration(self, t, state):
        intrinsic_acc = - self.stiffness*state[0] - self.damping*state[1]
        return array([state[1], self.acc[t / self.dt] + intrinsic_acc])

    def trajectory(self, x0, command):
        self.acc = self.basis.trajectory(command)
        # self.acc[-1] = self.acc[-2] # still due to ..utils.rk4 implementation
        t = 0.
        self.traj[0, :] = [x0[0], x0[1], self.acc[0]]
        i_t = 1
        state = x0
        while i_t < self.duration / self.dt:
            # print i_t, t, self.duration - self.dt
            t, state = rk4(t, self.dt, state, self.acceleration)
            # print state
            self.traj[i_t, :] = [state[0], state[1], self.acc[i_t]]
            i_t += 1
        return self.traj
