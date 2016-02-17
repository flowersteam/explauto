import numpy as np

from ..utils import rand_bounds
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist
from ..models.dataset import BufferedDataset as Dataset


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
                 k,
                 progress_mode):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.competence_measure = competence_measure
        self.win_size = win_size
        self.competence_mode = competence_mode
        self.dist_max = np.linalg.norm(self.bounds[0,:] - self.bounds[1,:])
        self.k = k
        self.progress_mode = progress_mode
        self.data_xc = Dataset(len(expl_dims), 1)
        self.data_sr = Dataset(len(expl_dims), 0)
        self.current_interest = 0.
              
            
    def add_xc(self, x, c):
        self.data_xc.add_xy(x, [c])
        
    def add_sr(self, x):
        self.data_sr.add_xy(x)
        
    def update_interest(self, i):
        self.current_interest += (1. / self.win_size) * (i - self.current_interest)

    def update(self, xy, ms, x=None):
        c = self.competence_measure(xy[self.expl_dims], ms[self.expl_dims], dist_max=self.dist_max)
        if self.progress_mode == 'local':
            if x is None:
                self.update_interest(self.interest_xc(xy[self.expl_dims], c))
            else:
                self.update_interest(self.interest_xc(x, c))
        elif self.progress_mode == 'global':
            pass
        else:
            raise NotImplementedError
        
        if x is None:
            self.add_xc(xy[self.expl_dims], c)
        else:
            self.add_xc(x, c)
        self.add_sr(ms[self.expl_dims])

    def n_points(self):
        return len(self.data_xc)
    
    def competence_global(self, mode='sw'):
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
        
    def mean_competence_pt(self, x):
        if self.n_points() > self.k: 
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            return np.mean([self.data_xc.get_y(idx) for idx in idxs])
        else:
            return self.competence()
                
    def interest_xc(self, x, c):
        """
        Interest of point x with competence c 
        
        """
        if self.n_points() > 0:
            idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
            sr_NN = self.data_sr.get_x(idx_sg_NN)
            c_old = self.competence_measure(x, sr_NN, dist_max=self.dist_max)
#             print 
#             print "x", x
#             print "sr_NN", sr_NN
#             print "c_old", c_old 
#             print "c_new", c
            return np.abs(c - c_old)
        else:
            return 0.
#         mean_local_comp = self.mean_competence_pt(x)
#         if mean_local_comp == 0:
#             return np.abs(c - mean_local_comp)
#         else:
#             return np.abs((c - mean_local_comp)/mean_local_comp)
        
    def interest_pt(self, x):
        if self.n_points() > self.k:
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            idxs = sorted(idxs)
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        else:
            return self.interest_global()
            
    def interest_global(self): 
        if self.n_points() < 2:
            return 0.
        else:
            idxs = range(self.n_points())[- self.win_size:]
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        
    def competence(self):
        return self.competence_global()
        
    def interest(self):
        if self.progress_mode == 'local':
            return self.current_interest
        elif self.progress_mode == 'global':
            return self.interest_global()
        else:
            raise NotImplementedError
        
        
interest_models = {'random': (RandomInterest, {'default': {}}),
                   'misc_random': (MiscRandomInterest, {'default': 
                       {'competence_measure': competence_dist,
                                   'win_size': 100,
                                   'competence_mode': 'knn',
                                   'k': 100,
                                   'progress_mode': 'local'}})}
