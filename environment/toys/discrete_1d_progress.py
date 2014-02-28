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
            s = 0
        elif m == self.s_card - 1:
            s = m
        elif m == 1:
            s = randint(self.s_card - 1)
        #else:
            #s = self.s_card - 1
        self.state[1] = s
