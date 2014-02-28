# -*- coding: utf-8 -*-

sys.path.append('../')
from environment.toys.discrete_1d_progress import *
from agent import Agent
from agent.config import get_config
from model.i_model import DiscreteProgressInterest
from model.competence import competence_bool
from experiment import Experiment

m_card = 3
s_card = 4
env = Discrete1dProgress(dict(m_ndims = 1, s_ndims = 1, m_card = m_card, s_card = s_card))

bounds = ((0, m_card), (0, s_card))

myconf = get_config(1, 1, bounds, ['discrete', dict(m_card=m_card, s_card=s_card)], ['discrete_progress', 'goal', dict(x_card=s_card, win_size=8)], competence_bool)

ag = Agent(**myconf)
expe_1 = Experiment(env, ag, 0, [0,1])

n_runs = 100
expe_1.run(n_runs)
