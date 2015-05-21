import numpy

from copy import deepcopy
from collections import deque

from ..utils.config import Space
from .competences import competence_exp, competence_dist  # TODO try without exp (now that we update on goal AND effect). Could solve the "interest for precision" problem
from ..utils import discrete_random_draw
from .interest_model import InterestModel
from math import ceil

class DiscretizedProgress(InterestModel):
    def __init__(self, conf, expl_dims, x_card, win_size, measure):
        InterestModel.__init__(self, expl_dims)
        self.conf = conf
        self.measure = measure
        # Add ceil to avoid having only 1 bin per dim for high dimensions (>=9 with x_card=400), which causes comp_min=comp_max and then a division by 0
        card = [int(ceil(x_card ** (1./len(expl_dims))))] * len(expl_dims)
        #print "CARD", x_card, len(expl_dims), card
        self.space = Space(numpy.hstack((conf.m_mins, conf.s_mins))[expl_dims],
                           numpy.hstack((conf.m_maxs, conf.s_maxs))[expl_dims], card)

        self.dist_min = numpy.sqrt(sum(self.space.bin_widths ** 2)) / 1.

        self.comp_max = measure(numpy.array([0.]), numpy.array([0.]), dist_min=self.dist_min)
        self.comp_min = measure(numpy.array([0.]), numpy.array([numpy.linalg.norm(conf.s_mins - conf.s_maxs)]), dist_min=self.dist_min)

        self.discrete_progress = DiscreteProgress(0, self.space.card,
                                                  win_size, measure, self.comp_min)


    def normalize_measure(self, measure):
        return (measure - self.comp_min)/(self.comp_max - self.comp_min)

    def sample(self):
        index = self.discrete_progress.sample(temp=self.space.card)[0]
        return self.space.rand_value(index).flatten()

    def update(self, xy, ms):
        measure = self.measure(xy, ms, dist_min=self.dist_min) # Either prediction error or competence error 
        x = xy[self.expl_dims]
        x_index = self.space.index(x)
        ms_expl = ms[self.expl_dims]
        ms_index = self.space.index(ms_expl)
        self.discrete_progress.queues[x_index].append(self.normalize_measure(measure))
        self.discrete_progress.queues[ms_index].append(self.normalize_measure(self.comp_max))


class DiscreteProgress(InterestModel):
    def __init__(self, expl_dims, x_card, win_size, measure, measure_init=0.):
        InterestModel.__init__(self, expl_dims)

        self.measure = measure
        self.win_size = win_size
        # self.t = [win_size] * self.xcard

        queue = deque([measure_init for t in range(win_size)], maxlen=win_size)
        self.queues = [deepcopy(queue) for _ in range(x_card)]

        # self.choices = numpy.zeros((10000, len(expl_dims)))
        # self.comps = numpy.zeros(10000)
        # self.t = 0

    def progress(self):
        return numpy.array([numpy.cov(zip(range(self.win_size), q), rowvar=0)[0, 1]
                            for q in self.queues])

    def sample(self, temp=3.):
        self.w = abs(self.progress())
        self.w = numpy.exp(temp * self.w - temp * self.w.max())  # / numpy.exp(3.)
        return discrete_random_draw(self.w)

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        self.queues[int(xy[self.expl_dims])].append(measure)
        # self.choices[self.t, :] = xy[self.expl_dims]
        # self.comps[self.t] = measure
        # self.t += 1


    def update_from_index_and_competence(self, index, competence):
        self.queues[index].append(competence)

interest_models = {'discretized_progress': (DiscretizedProgress,
                                            {'default': {'x_card': 400,
                                                         'win_size': 10,
                                                         'measure': competence_dist}}),
                   'discretized_progress_small': (DiscretizedProgress,
                                            {'default': {'x_card': 20,
                                                         'win_size': 10,
                                                         'measure': competence_dist}})
                   
                   }
                                             # 'comp_dist': {'x_card': 400,
                                                           # 'win_size': 10,
                                                           # 'measure': competence_dist}})}
