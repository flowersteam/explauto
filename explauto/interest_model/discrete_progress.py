import numpy

from copy import deepcopy
from collections import deque

from ..utils.config import Space
from .competences import competence_exp, competence_dist
from ..utils import discrete_random_draw
from .interest_model import InterestModel


class DiscretizedProgress(InterestModel):
    def __init__(self, conf, expl_dims, x_card, win_size, eps_random, measure):
        InterestModel.__init__(self, expl_dims)
        self.conf = conf
        self.measure = measure
        card = [int(x_card ** (1./len(expl_dims)))] * len(expl_dims)
        self.space = Space(numpy.hstack((conf.m_mins, conf.s_mins))[expl_dims],
                           numpy.hstack((conf.m_maxs, conf.s_maxs))[expl_dims], card)
        
        max_dist_in_cell = numpy.sqrt(sum(self.space.bin_widths ** 2))
        self.dist_min = max_dist_in_cell / 10.
        self.dist_max = max_dist_in_cell

        self.comp_max = measure(numpy.array([0.]), numpy.array([0.]), dist_min=self.dist_min)
        self.comp_min = measure(numpy.array([0.]), numpy.array([numpy.linalg.norm(conf.s_mins - conf.s_maxs)]), dist_max=self.dist_max)
        self.discrete_progress = DiscreteProgress(0, self.space.card, win_size, eps_random, measure)

    def normalize_measure(self, measure):
        return (measure - self.comp_min)/(self.comp_max - self.comp_min)

    def sample(self):
        index = self.discrete_progress.sample()[0]
        return self.space.rand_value(index).flatten()
    
    def sample_given_context(self, c, c_dims):
        '''
        Sample the region with max progress among regions that have the same context
            c: context value on c_dims dimensions
            c_dims: w.r.t sensory space dimensions
        '''
        index = self.discrete_progress.sample_given_context(c, c_dims, self.space)
        return self.space.rand_value(index).flatten()[list(set(range(len(self.space.cardinalities))) - set(c_dims))]

    def update(self, xy, ms):
        comp = self.measure(xy, ms, dist_min=self.dist_min, dist_max=self.dist_max)
        x = xy[self.expl_dims]
        x_index = self.space.index(x)
        ms_expl = ms[self.expl_dims]
        ms_index = self.space.index(ms_expl)
        
        # Only give competence if observed s is in the same cell as goal x
        # to avoid random fluctuations of progress due to random choices in the other cells and not to competence variations
        if ms_index == x_index:
            self.discrete_progress.update_from_index_and_competence(x_index, self.normalize_measure(comp))
            
        # Novelty bonus: if novel cell is reached, give it competence (= interest for win_size iterations)
        if sum([qi for qi in self.discrete_progress.queues[ms_index]]) == 0.:
            self.discrete_progress.update_from_index_and_competence(ms_index, self.normalize_measure(self.comp_max)) 


class DiscreteProgress(InterestModel):
    def __init__(self, expl_dims, x_card, win_size, eps_random, measure, measure_init=0.):
        InterestModel.__init__(self, expl_dims)

        self.measure = measure
        self.win_size = win_size
        self.eps_random = eps_random

        queue = deque([measure_init for _ in range(win_size)], maxlen=win_size)
        self.queues = [deepcopy(queue) for _ in range(x_card)]
        self.current_progress = numpy.zeros((x_card,))

    def progress(self):
        return numpy.array(self.current_progress)

    def sample(self):
        if numpy.random.random() < self.eps_random:
            # pick random cell
            return [numpy.random.randint(len(self.queues))]
        else:
            # pick with probability proportional to absolute progress
            self.w = abs(self.progress())
            if numpy.sum(self.w) > 0:
                self.w = self.w / numpy.sum(self.w)
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
                # Get the indices of the free dimensions of that region                           
                multi_new = tuple(numpy.array(list(multi_old))[free_dims])
                # Get progress in that region
                p = self.current_progress[i]
                progress_array[multi_new] = p
        # Choose the region with max progress
        self.w = abs(progress_array)
        if numpy.sum(self.w) > 0:
            self.w = self.w / numpy.sum(self.w)
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
        self.current_progress[index] = numpy.mean([self.queues[index][i] for i in range(0,self.win_size/2)]) - numpy.mean([self.queues[index][i] for i in range(self.win_size/2,self.win_size)])
                            


interest_models = {'discretized_progress': (DiscretizedProgress,
                                            {'default': {'x_card': 400,
                                                         'win_size': 10,
                                                         'eps_random': 0.3,
                                                         'measure': competence_dist}})}
