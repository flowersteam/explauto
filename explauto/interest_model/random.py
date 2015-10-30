import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from ..utils import rand_bounds
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist
from models.dataset import IncrementalBufferedDataset as Dataset


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
                 competence_k,
                 progress_mode):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.competence_measure = competence_measure
        self.win_size = win_size
        self.competence_mode = competence_mode
        self.competence_k = competence_k
        self.progress_mode = progress_mode
        self.data_xc = Dataset(len(expl_dims), 1)
        self.current_interest = 0.
        
        
        
#     def add_x(self, x):
#         if self.data_x is None:
#             self.data_x = np.array([x])
#         else:
#             self.data_x = np.append(self.data_x, np.array([x]), axis=0)    
#         
#     def add_c(self, c):
#         if self.data_c is None:
#             self.data_c = np.array([c])
#         else:
#             self.data_c = np.append(self.data_c, c) 
            
            
    def add_xc(self, x, c):
        self.data_xc.add_xy(x, [c])
        
#     def add_i(self, i):
#         if self.data_i is None:
#             self.data_i = np.array([i])
#         else:
#             self.data_i = np.append(self.data_i, i)     

    def update_interest(self, i):
        self.current_interest += (1. / self.win_size) *(i - self.current_interest) 

    def update(self, xy, ms):
        #self.add_x(xy[self.expl_dims])
        #print
        #print "competence_measure ", "xy=", len(xy), "ms=", len(ms)
        c = self.competence_measure(xy, ms)
        #print "competence_measure ", "xy=", xy, "ms=", ms, "c=", c
#         print "self.expl_dims", self.expl_dims
#         print "xy", xy
#         print "ms", ms
#         print "c", c
#         print 
        #self.add_c(c)
        self.update_interest(self.interest_pt(xy[self.expl_dims], c))
        self.add_xc(xy[self.expl_dims], c)

    def n_points(self):
#         if self.data_x is None:
#             return 0
#         else:
#             return np.shape(self.data_x)[0]
        return len(self.data_xc)
    
    def competence(self, mode='sw'):
        if self.n_points() > 0:
            if mode == 'all':
                return np.mean(self.data_c)
            elif mode == 'sw':
                idxs = range(self.n_points())[- self.win_size:]
                return np.mean([self.data_xc.get_y(idx) for idx in idxs])
            else:
                raise NotImplementedError
        else:
            return 0.
        
    def competence_pt(self, x, mode=None):
        mode = mode or self.competence_mode
        if mode == 'knn':
            if self.n_points() > self.competence_k:
#                 assert len(x) == len(self.data_x[0]), (len(x), len(self.data_x[0]))
#                 weights = 'uniform'
#                 knr = KNeighborsRegressor(n_neighbors=self.competence_k, 
#                                           weights=weights)
#                 knr.fit(self.data_x, self.data_c)
#                 return knr.predict(x) 
                dists, idxs = self.data_xc.nn_x(x, k=self.competence_k)
                #print "competences around x=", x, ": dists=", dists, "competences=", [self.data_xc.get_y(idx)[0] for idx in idxs]
                return np.mean([self.data_xc.get_y(idx) for idx in idxs])
            else:
                return self.competence()
        else:
            return self.competence(mode=mode)
                
    def interest_pt(self, x, c):
        """
        Interest of point x with competence c with respect to local competence
        
        """
        mean_local_comp = self.competence_pt(x)
        #print "mean local comp=", mean_local_comp, "c=", c, "i=", np.abs(c - mean_local_comp)    
        return np.abs(c - mean_local_comp)        
        
    def interest_local(self): 
#         if self.n_points() < 2:
#             return 0.
#         else:   
#             idxs = range(self.n_points())[- self.win_size:]
#             return np.mean(self.data_i[idxs])
        return self.current_interest
        
    def interest_global(self): 
        if self.n_points() < 2:
            return 0.
        else:
            idxs = range(self.n_points())[- self.win_size:]
            v = self.data_c[idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        
    def interest(self):
        if self.progress_mode == 'local':
            return self.interest_local()
        elif self.progress_mode == 'global':
            return self.interest_global()
        else:
            raise NotImplementedError
        
        
interest_models = {'random': (RandomInterest, {'default': {}}),
                   'miscRandom': (MiscRandomInterest, {'default': 
                       {'competence_measure': lambda target,reached : competence_exp(target, reached, dist_min=0.0, power=1.),
                                   'win_size': 100,
                                   'competence_mode': 'knn',
                                   'competence_k': 5,
                                   'progress_mode': 'local'}})}
