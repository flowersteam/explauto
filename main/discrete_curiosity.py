# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

sys.path.append('../')
from environment.toys.discrete_1d_progress import *
from agent import Agent
from agent.config import get_config
from model.i_model import DiscreteProgressInterest
from model.competence import competence_bool
from experiment import Experiment

# <codecell>

m_card = 7
s_card = 7
env = Discrete1dProgress(dict(m_ndims = 1, s_ndims = 1, m_card = m_card, s_card = s_card))

# <codecell>

mean(numpy.random.randint(2, size = 1000))

# <codecell>

myconf = get_config(1, 1, ['discrete', dict(m_card=m_card, s_card=s_card, lambd = 0.01)], ['discrete_progress', 'goal', dict(x_card=s_card, win_size=10, measure = competence_bool)])

# <codecell>

ag = Agent(**myconf)
expe_1 = Experiment(env, ag, 0, [0,1])

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

# <codecell>

2**8

# <codecell>

figure()
win = 40
spent = []
for t in range(expe_1.i_rec-40):
    spent.append([float(sum(expe_1.records[t:t+win, 1] == s))/win * 100 for s in range(s_card)])
plot(spent)
legend([str(i) for i in range(s_card)])    

# <codecell>

x = linspace(0., 1., 100)
plot(x, exp(2. * x) / exp(2.))

# <codecell>

zip(range(4), x)

# <codecell>

n_trials = 10
logs_x = zeros((n_trials, s_card))
for trial in range(n_trials):
    del ag, expe_1
    ag = Agent(**myconf)
    expe_1 = Experiment(env, ag, 0, [0,1])
    expe_1.run(10)
    logs_x[trial, :] = [(ag.choices[:ag.t, :] == x).sum() for x in range(s_card)]

# <codecell>

logs_x.sum(axis=0)

# <codecell>

logs_x

# <codecell>

env.s_card

# <codecell>


# <codecell>

print ag.choices[:ag.t, :]

# <codecell>

[(ag.choices[:ag.t, :] == x).sum() for x in range(s_card)]

# <codecell>

from model.sm_model import LidstoneModel

# <codecell>

model = LidstoneModel(3, 4)

# <codecell>

model.counts

# <codecell>

model.update(2, 1)

# <codecell>

p

# <codecell>

from collections import deque

# <codecell>

q = deque(zeros(3), 3)

# <codecell>

q.append(array([2., 0.]))

# <codecell>

print q

# <codecell>

cov(array(q), rowvar = 0)

# <codecell>

d = deque([[t, 0.] for t in range(4)])
print d

# <codecell>

cov(d, rowvar=0)

# <codecell>

[0, 1] == 0

# <codecell>

d = array([0., 0., 0.])
d = d / d.sum()

# <codecell>

(discrete_random_draw(d, nb=10000) == 2).sum()

# <codecell>

d

# <codecell>

d = array([1, 1, 1])

# <codecell>

d

# <codecell>


