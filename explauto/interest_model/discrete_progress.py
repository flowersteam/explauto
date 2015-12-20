import numpy

from copy import deepcopy
from collections import deque

from ..utils.config import Space
from .competences import competence_exp, competence_dist  # TODO try without exp (now that we update on goal AND effect). Could solve the "interest for precision" problem
from ..utils import discrete_random_draw
from .interest_model import InterestModel


class DiscretizedProgress(InterestModel):
    def __init__(self, conf, expl_dims, x_card, win_size, measure):
        InterestModel.__init__(self, expl_dims)
        self.conf = conf
        self.measure = measure
        card = [int(x_card ** (1./len(expl_dims)))] * len(expl_dims)
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
    
    def sample_given_context(self, c, c_dims):
        '''
        Sample the region with max progress among regions that have the same context
            c: context value on c_dims dimensions
            c_dims: w.r.t sensori space dimensions
        '''
        index = self.discrete_progress.sample_given_context(c, c_dims, self.space)
        return self.space.rand_value(index).flatten()[list(set(range(len(self.space.cardinalities))) - set(c_dims))]

    def update(self, xy, ms):
        measure = self.measure(xy, ms, dist_min=self.dist_min)
        x = xy[self.expl_dims]
        x_index = self.space.index(x)
        self.discrete_progress.update_from_index_and_competence(x_index, self.normalize_measure(measure))
        #ms_expl = ms[self.expl_dims]
        #ms_index = self.space.index(ms_expl)
        #self.discrete_progress.update_from_index_and_competence(ms_index, self.normalize_measure(self.comp_max)) # Suitable only in deterministic environments


class DiscreteProgress(InterestModel):
    def __init__(self, expl_dims, x_card, win_size, measure, measure_init=0.):
        InterestModel.__init__(self, expl_dims)

        self.measure = measure
        self.win_size = win_size

        queue = deque([measure_init for t in range(win_size)], maxlen=win_size)
        self.queues = [deepcopy(queue) for _ in range(x_card)]

    def progress(self):
        return numpy.array([numpy.cov(zip(range(self.win_size), q), rowvar=0)[0, 1]
                            for q in self.queues])

    def sample(self, temp=3.):
        self.w = abs(self.progress())
        self.w = numpy.exp(temp * self.w - temp * self.w.max())  # / numpy.exp(3.)
        return discrete_random_draw(self.w)
    
    def sample_given_context(self, c, c_dims, space):
        free_dims = list(set(range(len(space.cardinalities))) - set(c_dims))
        free_cardinalities = tuple(numpy.array(list(space.cardinalities))[free_dims])
        progress_array = numpy.zeros(free_cardinalities)
        
        # Get index of context on context dimensions as multi_context
        value = numpy.zeros(len(space.cardinalities))
        value[c_dims] = c
        multi_context = tuple(space.discretize(value, c_dims)) 
        
        for i in range(len(self.queues)):
            multi_old = space.index2multi(i)
            # if that region is included in the context 
            if tuple(numpy.array(list(multi_old))[c_dims]) == multi_context:
                # Get the indices of the free dimensions of that regions                                
                multi_new = tuple(numpy.array(list(multi_old))[free_dims])
                # Get the last competences in that region
                q = self.queues[i]
                # Compute progress in that region
                p = numpy.cov(zip(range(self.win_size), q), rowvar=0)[0, 1]
                progress_array[multi_new] = p
        # Choose the region with max progress
        self.w = abs(progress_array)
        temp = 3.
        self.w = numpy.exp(temp * self.w - temp * self.w.max())
        index_new = discrete_random_draw(self.w.flatten())
        # Convert the index of the region from the free dims to all dims
        multi_new = numpy.unravel_index(index_new, free_cardinalities)
        multi_old = numpy.zeros(len(space.cardinalities), dtype=numpy.int64)
        multi_old[c_dims] = multi_context
        multi_old[free_dims] = list(multi_new)
        index = space.multi2index(tuple(multi_old))
        return index
        
    def update(self, xy, ms):
        raise NotImplementedError
    
    def update_from_index_and_competence(self, index, competence):
        self.queues[index].append(competence)

interest_models = {'discretized_progress': (DiscretizedProgress,
                                            {'default': {'x_card': 400,
                                                         'win_size': 10,
                                                         'measure': competence_dist}})}
