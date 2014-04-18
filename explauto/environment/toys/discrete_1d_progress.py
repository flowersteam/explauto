from numpy.random import randint

from ..environment import Environment


class Discrete1dProgress(Environment):
    def __init__(self, **kwargs):
        Environment.__init__(self, ndims=2)

        self.m_card = kwargs['m_card']
        self.s_card = kwargs['s_card']
        self.writable = [0]
        self.readable = [0, 1]

    def next_state(self, ag_state):
        m = ag_state

        # if m == 0:
        #     s = 2
        # elif m == 1:
        #     s = m
        # elif m == 2:
        #     s = randint(2)

        if m == 0:
            s = 0
        elif m == 1:
            s = randint(2)
        elif m <= 4:
            s = 2
        elif m == 5:
            s = 3
        elif m == 6:
            s = randint(2) + 4

        self.state[0] = m
        self.state[1] = s
