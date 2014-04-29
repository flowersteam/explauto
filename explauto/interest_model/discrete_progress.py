import numpy

from copy import deepcopy
from collections import deque

from .competences import competence_exp
from ..utils import discrete_random_draw
from .interest_model import InterestModel


class DiscreteProgressInterest(InterestModel):
    def __init__(self, i_dims, x_card, win_size, measure):
        InterestModel.__init__(self, i_dims)

        self.measure = measure
        self.win_size = win_size
        # self.t = [win_size] * self.xcard

        queue = deque([0. for t in range(win_size)], maxlen=win_size)
        self.queues = [deepcopy(queue) for _ in range(x_card)]

        self.choices = numpy.zeros((10000, len(i_dims)))
        self.comps = numpy.zeros(10000)
        self.t = 0

    def progress(self):
        return numpy.array([numpy.cov(zip(range(self.win_size), q), rowvar=0)[0, 1]
                            for q in self.queues])

    def sample(self):
        w = abs(self.progress())
        w = numpy.exp(3. * w) / numpy.exp(3.)
        return discrete_random_draw(w)

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        self.queues[int(xy[self.i_dims])].append(measure)
        self.choices[self.t, :] = xy[self.i_dims]
        self.comps[self.t] = measure
        self.t += 1


interest_models = {'discrete_progress': (DiscreteProgressInterest,
                                         {'default': {'x_card': 10,
                                                      'win_size': 10,
                                                      'measure': competence_exp}})}
