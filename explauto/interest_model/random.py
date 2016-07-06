import numpy as np

from ..utils import rand_bounds
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist
from ..models.dataset import BufferedDataset as Dataset


class RandomInterest(InterestModel):
    def __init__(self, conf, expl_dims):
        InterestModel.__init__(self, expl_dims)

        self.bounds = conf.bounds[:, expl_dims]
        self.ndims = self.bounds.shape[1]

    def sample(self):
        return rand_bounds(self.bounds).flatten()

    def update(self, xy, ms):
        pass

    def sample_given_context(self, c, c_dims):
        '''
        Sample randomly on dimensions not in context
            c: context value on c_dims dimensions, not used
            c_dims: w.r.t sensori space dimensions
        '''
        return self.sample()[list(set(range(self.ndims)) - set(c_dims))]


class MiscRandomInterest(RandomInterest):
    """
    Add some features to the RandomInterest random babbling class.
    
    Allows to query the recent interest and competence.
    Interest computation is adapted from IROS paper 
    'Modular Active Curiosity-Driven Discovery of Tool Use',
    without taking the absolute value at each iteration.
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims,
                 competence_measure,
                 win_size):

        RandomInterest.__init__(self, conf, expl_dims)

        self.competence_measure = competence_measure
        self.win_size = win_size
        self.dist_max = np.linalg.norm(self.bounds[0,:] - self.bounds[1,:])
        self.data_xc = Dataset(len(expl_dims), 1)
        self.data_sr = Dataset(len(expl_dims), 0)
        self.current_progress = 0.
        self.current_interest = 0.


    def add_xc(self, x, c):
        self.data_xc.add_xy(x, [c])

    def add_sr(self, x):
        self.data_sr.add_xy(x)

    def update_interest(self, i):
        self.current_progress += (1. / self.win_size) * (i - self.current_progress)
        self.current_interest = abs(self.current_progress)

    def update(self, xy, ms, snnp=None, sp=None):
        c = self.competence_measure(xy[self.expl_dims], ms[self.expl_dims], dist_max=self.dist_max)
        interest = self.interest_xc(xy[self.expl_dims], c)
        self.update_interest(interest)
        self.add_xc(xy[self.expl_dims], c)
        self.add_sr(ms[self.expl_dims])
        return interest

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

    def interest_xc(self, x, c):
        if self.n_points() > 0:
            idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
            sr_NN = self.data_sr.get_x(idx_sg_NN)
            c_old = competence_dist(x, sr_NN, dist_max=self.dist_max)
            return c - c_old
            #return np.abs(c - c_old)
        else:
            return 0.

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

    def competence(self): return self.competence_global()

    def interest(self): return self.current_interest



interest_models = {'random': (RandomInterest, {'default': {}}),
                   'misc_random': (MiscRandomInterest, {'default': 
                                   {'competence_measure': competence_dist,
                                    'win_size': 100}})}

