# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
from numpy import zeros
import sys
sys.path.append('../')
from explauto.environment.toys.discrete_1d_progress import Discrete1dProgress
from explauto.agent import Agent, get_config
from explauto.interest_models.competences import competence_bool
from explauto.experiment import Experiment

# <codecell>

m_card = 7
s_card = 7
env = Discrete1dProgress(dict(m_card = m_card, s_card = s_card))

# <codecell>

myconf = get_config(1, 1, ['discrete', dict(m_card=m_card, s_card=s_card, lambd = 0.01)], ['discrete_progress', 'goal', dict(x_card=s_card, win_size=10, measure = competence_bool)])

# <codecell>

ag = Agent(**myconf)
expe_1 = Experiment(env, ag)

# <codecell>

n_trials = 200
progr = zeros((n_trials, s_card))
for i in range(n_trials):
    expe_1.run()
    progr[i,:] = ag.i_model.progress()

# <codecell>

clf()
plot(progr)
legend([str(i) for i in range(s_card)])
[text(n_trials * 1.1, progr[-1, x], str(x)) for x in range(s_card)]    

# <codecell>

figure()
for s in range(s_card):
    inds = nonzero(ag.i_model.choices.flatten()[:ag.i_model.t] == s)[0]
    plot(inds, ag.i_model.comps[inds], '*', ms=12)
legend([str(i) for i in range(s_card)])    
