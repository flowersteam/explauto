
from ..environment import Environment
from numpy.random import rand
from functools import partial

f = lambda n, temp: 1./(1. + n / temp)



class HierarchicalMild(Environment):
    def __init__(self, config_dict):
        Environment.__init__(self, config_dict)
        #self.m_card = config_dict['m_card']
        #self.s_card = config_dict['s_card']
        temperatures = [1.] * self.n_sdims
        self.ressource_fun = [partial(f, temp = temp) for _, temp in enumerate(temperatures)]
        self.n_used = [0] * self.n_sdims
    def next_state(self):
        m = int(self.state[0])
        s = self.state[-self.n_sdims:]
        ressource_available = rand() < self.ressource_fun[m](self.n_used[m])
        self.n_used[m] += 1
        if m > 0:
            self.n_used[m-1] = 0
        s[m] = ressource_available
        self.state[-self.n_sdims:] = s
