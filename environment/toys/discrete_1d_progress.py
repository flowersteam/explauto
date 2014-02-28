from numpy.random import randint
from environment import Environment

class Discrete1dProgress(Environment):
    def __init__(self, config_dict):
        Environment.__init__(self, config_dict)
        self.m_card = config_dict['m_card']
        self.s_card = config_dict['s_card']
    def next_state(self):
        m = self.state[0]
        if m == 0:
            s = randint(self.s_card)
        elif m == 1 or m == 2:
            s = m + 1
        else:
            s = self.s_card - 1
        self.state[1] = s
