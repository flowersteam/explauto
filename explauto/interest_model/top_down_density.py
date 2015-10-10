import numpy as np

from ..utils import bounds_min_max, rand_bounds
from .interest_model import InterestModel
from sklearn.neighbors import KernelDensity


class TDDensityInterest(InterestModel):
    def __init__(self, conf, expl_dims, kernel='tophat', bandwith=0.1):
        InterestModel.__init__(self, expl_dims)

        self.bounds = conf.bounds[:, expl_dims]
        self.kernel = kernel
        self.bandwith = bandwith
        self.data_s = []
        self.td_density = None
        self.up_to_date = False
    
    
    def check_bounds(self, s):
        return bounds_min_max(s, self.bounds[0], self.bounds[1])

    def random_sample(self):
        return rand_bounds(self.bounds).flatten()
        
    def sample(self):
        if self.n_td_points() == 0:
            return self.random_sample()        
        elif self.up_to_date:
            return self.check_bounds(self.td_density.sample(n_samples=1))
        else:
            self.update_density()
            return self.sample()
    
    def n_td_points(self):
        return len(self.data_s)
    
    def top_down_goal(self, s):
        self.up_to_date = False
        self.data_s.append(s)        
        
    def update_density(self, s):
        self.td_tree = KernelDensity(kernel=self.kernel, bandwith=self.bandwith).fit(np.array(self.data_s))
        self.up_to_date = True
        
    def update(self, xy, ms):
        pass

        
interest_models = {'TDDensity': (TDDensityInterest, {'default': {'kernel': 'tophat',
                                                                 'bandwith': 0.1}})}
