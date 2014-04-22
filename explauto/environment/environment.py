from numpy import zeros

from ..utils.config import Configuration
from ..utils.observer import Observable


class Environment(Observable):
    """ Abstract class to define environments.
        :param array m_mins, m_maxs, s_mins, s_maxs: bounds of the motor (m) and sensory (s) spaces
    """
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):
        Observable.__init__(self)

        self.conf = Configuration(m_mins, m_maxs, s_mins, s_maxs)
        self.state = zeros(self.conf.ndims)

    def update(self, ag_state):
        m = self.compute_motor_command(ag_state)
        self.state[:self.conf.m_ndims] = m
        self.emit('motor', m)

        s = self.compute_sensori_effect()
        self.state[-self.conf.s_ndims:] = s
        self.emit('sensori', s)

    def compute_motor_command(self, ag_state):
        raise NotImplementedError

    def compute_sensori_effect(self):
        raise NotImplementedError

    # def post_processing(self):
    #     self.state = minimum(self.state, self.bounds[:,1])
    #     self.state = maximum(self.state, self.bounds[:,0])

    def read(self):
        return self.state[self.readable]

    # def write(self, data):
    #     self.state[self.writable] = data

    def dataset(self, orders):
        n = orders.shape[0]
        m_ndims = orders.shape[1]

        data = zeros((n, self.conf.ndims))
        data[:, :m_ndims] = orders

        for i, m in enumerate(orders):
            self.next_state(m)
            data[i, m_ndims:] = self.state[m_ndims:]

        return data
