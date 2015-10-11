import numpy as np

from sklearn.neighbors import KernelDensity

from ..utils import bounds_min_max, rand_bounds
from .interest_model import InterestModel
from .tree import InterestTree, Tree
import tree.interest_models as tree_interest_models


class TDDensityInterest(InterestModel):
    def __init__(self, conf, expl_dims, kernel, bandwith):
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
    
    def add_top_down_goal(self, s):
        self.up_to_date = False
        self.data_s.append(s)        
        
    def update_density(self, s):
        self.td_tree = KernelDensity(kernel=self.kernel, bandwith=self.bandwith).fit(np.array(self.data_s))
        self.up_to_date = True
        
    def update(self, xy, ms):
        pass


class TDDensityTree(Tree):
    def __init__(self, 
                 get_data_x, 
                 bounds_x, 
                 get_data_c, 
                 set_data_c, 
                 max_points_per_region, 
                 max_depth,
                 split_mode, 
                 competence_measure,
                 progress_win_size, 
                 progress_measure, 
                 sampling_mode, 
                 comp_min_cut,
                 idxs=[], 
                 split_dim=0):
        
        Tree.__init__(self,
                     get_data_x, 
                     bounds_x, 
                     get_data_c, 
                     set_data_c, 
                     max_points_per_region, 
                     max_depth,
                     split_mode, 
                     competence_measure,
                     progress_win_size, 
                     progress_measure, 
                     sampling_mode, 
                     comp_min_cut,
                     idxs=[], 
                     split_dim=0)
        
        self.data_s = None
        
    
    def sample_softmax(self, temperature=1.):
        pass
    
    def split(self):
        Tree.split(self)
        self.split_value
        
    
class TDDensityTreeInterest(InterestTree):
    def __init__(self, 
                 conf, 
                 expl_dims, 
                 kernel, 
                 bandwith):
        
        InterestTree.__init__(self, 
                              conf, 
                              expl_dims,
                              **tree_interest_models['tree'][1]['default']
                              )

        self.bounds = conf.bounds[:, expl_dims]
        self.kernel = kernel
        self.bandwith = bandwith
        self.data_s = []  
        
    def create_tree(self):
        self.tree = TDDensityTree(lambda:self.data_x, 
                                 np.array(self.bounds, dtype=np.float), 
                                 lambda:self.data_c,
                                 lambda idx, c: self.set_data_c(idx, c), 
                                 max_points_per_region=self.max_points_per_region, 
                                 max_depth=self.max_depth,
                                 split_mode=self.split_mode, 
                                 competence_measure=self.competence_measure,
                                 progress_win_size=self.progress_win_size, 
                                 progress_measure=self.progress_measure, 
                                 sampling_mode=self.sampling_mode,
                                 comp_min_cut=self.comp_min_cut,
                                 idxs=[])
        
    def sample(self):
        if self.n_td_points() == 0:
            return InterestTree.sample(self)        
        else:
            return self.check_bounds(self.td_density.sample(n_samples=1))
    
    def n_td_points(self):
        return len(self.data_s)
    
    def add_top_down_goal(self, s):
        self.data_s.append(s)        
        
        
        
        
interest_models = {'TDDensity': (TDDensityInterest, {'default': {'kernel': 'tophat',
                                                                 'bandwith': 0.1}})}
