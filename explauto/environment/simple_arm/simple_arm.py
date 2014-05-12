import numpy as np

from .. import Environment
from ...utils import bounds_min_max


def forward(angles, lengths):
    """ Link object as defined by the standard DH representation.
    :param list angles: angles of each joint
    :param list lengths: length of each segment

    .. warning:: angles and lengths should be the same size.
    """

    if len(angles) != len(lengths):
        raise ValueError('angles and lengths must be the same size!')

    a = np.array(angles)
    a = np.cumsum(a)
    return sum(np.cos(a)*lengths), sum(np.sin(a)*lengths)


def lengths(n_dofs, ratio):
    l = np.ones(n_dofs)
    for i in range(1, n_dofs - 1):
        l[i] = l[i-1] / ratio
    return l / sum(l)


class SimpleArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, noise):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length_ratio = length_ratio
        self.noise = noise

        self.lengths = lengths(self.conf.m_ndims, self.length_ratio)
        self.readable = range(self.conf.m_ndims + self.conf.s_ndims)
        self.writable = range(self.conf.m_ndims)

    def compute_motor_command(self, ag_state):
        motor_cmd = ag_state
        return bounds_min_max(motor_cmd, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self):
        res = np.array(forward(self.state[:self.conf.m_ndims], self.lengths))
        res += self.noise * np.random.randn(*res.shape)
        return res
