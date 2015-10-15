import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from ..utils import rand_bounds
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist


class RandomInterest(InterestModel):
    def __init__(self, conf, expl_dims):
        InterestModel.__init__(self, expl_dims)

        self.bounds = conf.bounds[:, expl_dims]

    def sample(self):
        return rand_bounds(self.bounds).flatten()

    def update(self, xy, ms):
        pass


class MiscRandomInterest(RandomInterest):
    """
    Add some features to the RandomInterest random babbling class.
    
    Allows to query the recent interest in the whole space,
    the recent competence on the babbled points in the whole space, 
    the competence around a given point based on a mean of the knns.   
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims,
                 competence_measure,
                 win_size,
                 competence_mode,
                 competence_k):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.competence_measure = competence_measure
        self.win_size = win_size
        self.competence_mode = competence_mode
        self.competence_k = competence_k
        self.data_x = None
        self.data_c = None
        
        
    def add_x(self, x):
        if self.data_x is None:
            self.data_x = np.array([x])
        else:
            self.data_x = np.append(self.data_x, np.array([x]), axis=0)    
        
    def add_c(self, c):
        if self.data_c is None:
            self.data_c = np.array([c])
        else:
            self.data_c = np.append(self.data_c, c)     

    def update(self, xy, ms):
        self.add_x(xy[self.expl_dims])
        c = self.competence_measure(xy, ms)
        self.add_c(c)

    def n_points(self):
        if self.data_x is None:
            return 0
        else:
            return np.shape(self.data_x)[0]
    
    def competence(self, mode='sw'):
        if self.n_points() > 0:
            if mode == 'all':
                return np.mean(self.data_c)
            elif mode == 'sw':
                idxs = range(self.n_points())[- self.win_size:]
                return np.mean(self.data_c[idxs])
            else:
                raise NotImplementedError
        else:
            return 0
        
    def competence_pt(self, x, mode=None):
        mode = mode or self.competence_mode
        if mode == 'knn':
            if self.n_points() > self.competence_k:
                weights = 'uniform'
                knr = KNeighborsRegressor(n_neighbors=self.competence_k, 
                                          weights=weights)
                knr.fit(self.data_x, self.data_c)
                return knr.predict(x) 
            else:
                return self.competence()
        else:
            return self.competence(mode=mode)
                
    def interest(self):
        if self.n_points() < 2:
            return 0
        else:
            idxs = range(self.n_points())[- self.win_size:]
            v = self.data_c[idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        
        
        
interest_models = {'random': (RandomInterest, {'default': {}}),
                   'miscRandom': (MiscRandomInterest, {'default': 
                       {'competence_measure': lambda target,reached : competence_exp(target, reached, 0.01, 0.1, 1.),
                                   'win_size': 20,
                                   'competence_mode': 'knn',
                                   'competence_k': 20}})}
