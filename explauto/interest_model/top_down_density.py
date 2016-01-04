from sklearn.neighbors import KernelDensity

import numpy as np

from ..utils import bounds_min_max, rand_bounds
from .interest_model import interest_models as all_interest_models
from .random import MiscRandomInterest
from .tree import InterestTree, Tree


class TDDensityInterest(MiscRandomInterest):
    def __init__(self, conf, expl_dims, kernel, bandwidth, time_window, self_weight, td_weight):
        MiscRandomInterest.__init__(self, conf, expl_dims, **all_interest_models['miscRandom_local'][1]['default'])
        self.bounds = conf.bounds[:, expl_dims]
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.time_window = time_window
        self.self_weight = self_weight
        self.td_weight = td_weight
        self.data_s = []
        self.td_tree = None
        self.up_to_date = False    
    
    def check_bounds(self, s):
        return bounds_min_max(s, self.bounds[0], self.bounds[1])

    def random_sample(self):
        return rand_bounds(self.bounds).flatten()
        
    def sample(self):
        """
        We sample with td_points density distribution with some probability, and otherwise randomly.
        
        """
        if self.n_td_points() == 0:
            return self.random_sample()        
        elif self.up_to_date:
            if self.td_weight / (self.self_weight + self.td_weight) > np.random.rand():
                s = self.check_bounds(self.td_tree.sample(n_samples=1))
                return self.check_bounds(s)
            else:
                return self.random_sample()
        else:
            self.update_density()
            return self.sample()
    
    def n_td_points(self):
        return len(self.data_s)
    
    def add_top_down_goal(self, s):
        self.up_to_date = False
        self.data_s.append(s)        
        
    def update_density(self):
        self.td_tree = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(np.array(self.data_s[-self.time_window:]))
        self.up_to_date = True
        

class TDDensityTree(Tree):
    def __init__(self, 
                 get_data_x, 
                 bounds_x, 
                 get_data_c, 
                 get_data_s,
                 get_td_time_limit,
                 self_weight,
                 td_weight,
                 set_data_c, 
                 max_points_per_region, 
                 max_depth,
                 split_mode, 
                 competence_measure,
                 progress_win_size, 
                 progress_measure, 
                 interest_measure, 
                 sampling_mode, 
                 comp_min_cut,
                 idxs=None, 
                 split_dim=0):
        
        self.get_data_s = get_data_s
        self.get_td_time_limit = get_td_time_limit
        self.self_weight = self_weight
        self.td_weight = td_weight
        self.td_idxs = []
        
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
                     interest_measure, 
                     sampling_mode, 
                     comp_min_cut,
                     idxs=idxs, 
                     split_dim=split_dim)
        
        
    def create_subtrees(self, l_bounds_x, g_bounds_x, lower_idx, greater_idx, split_dim):
        self.lower = TDDensityTree(self.get_data_x, 
                                  l_bounds_x, 
                                  self.get_data_c, 
                                  self.get_data_s,
                                  self.get_td_time_limit,
                                  self.self_weight,
                                  self.td_weight,
                                  self.set_data_c, 
                                  self.max_points_per_region, 
                                  self.max_depth - 1,
                                  self.split_mode,                           
                                  self.competence_measure,
                                  self.progress_win_size, 
                                  self.progress_measure, 
                                  self.interest_measure, 
                                  self.sampling_mode, 
                                  self.comp_min_cut,
                                  idxs=lower_idx, 
                                  split_dim=split_dim)
        
        self.greater = TDDensityTree(self.get_data_x, 
                                    g_bounds_x, 
                                    self.get_data_c, 
                                    self.get_data_s,  
                                    self.get_td_time_limit,
                                    self.self_weight,
                                    self.td_weight,
                                    self.set_data_c, 
                                    self.max_points_per_region, 
                                    self.max_depth - 1,
                                    self.split_mode,                    
                                    self.competence_measure,
                                    self.progress_win_size, 
                                    self.progress_measure, 
                                    self.interest_measure, 
                                    self.sampling_mode, 
                                    self.comp_min_cut,
                                    idxs=greater_idx, 
                                    split_dim=split_dim)   
        
        for td_idx in self.td_idxs:
            if self.get_data_s()[td_idx, self.split_dim] > self.split_value:
                self.greater.td_idxs.append(td_idx)
            else:
                self.lower.td_idxs.append(td_idx)
    
    def compute_interest(self):
        #print "--------leaf", self.bounds_x, "progress", self.progress, "td_interest", self.td_interest()
        self.max_interest = self.progress * self.volume() * (self.self_weight + self.td_interest())
        
    def td_interest(self):
        self.update_td_idxs()
        return self.td_weight * float(len(self.td_idxs)) #self.td_weight already divided by time_window
        
    def update_td_idxs(self):
        self.td_idxs = [td_idx for td_idx in self.td_idxs if td_idx >= self.get_td_time_limit()]
        
    def add_top_down_goal(self, td_idx):
        self.pt2leaf(self.get_data_s()[td_idx, :]).td_idxs.append(td_idx)
        self.recompute_tree_max_interest()
    
    def print_tree(self, depth=0):
        """ 
        Print human-readable tree (recursive).
        
        """
        print
        for _ in range(depth):
            print "    ",
        print "Node bounds:", self.bounds_x 
        if self.leafnode:
            for _ in range(depth):
                print "    ",
            print "Leaf progress:", self.progress, "td_interest", self.td_interest(), "max interest", self.max_interest
            for _ in range(depth):
                print "    ",
            print "Leaf indices    :", self.idxs
            for _ in range(depth):
                print "    ",
            print "Leaf points     :", self.get_data_x()[self.idxs][:,0]
            for _ in range(depth):
                print "    ",
            print "Leaf td indices    :", self.td_idxs
            for _ in range(depth):
                print "    ",
            if self.get_data_s() is not None:
                print "Leaf td points     :", self.get_data_s()[self.td_idxs][:,0]
                for _ in range(depth):
                    print "    ",
            print "Leaf competences:", self.get_data_c()[self.idxs]
        else:
            self.lower.print_tree(depth+1)
            self.greater.print_tree(depth+1)
    
    
class TDDensityTreeInterest(InterestTree):
    def __init__(self, conf, expl_dims, time_window, self_weight, td_weight):
        
        self.data_s = None
        self.time_window = time_window
        self.self_weight = self_weight
        self.td_weight = td_weight
        
        InterestTree.__init__(self, 
                              conf, 
                              expl_dims,
                              **all_interest_models['tree'][1]['default']
                              )
        
    def create_tree(self):
        self.tree = TDDensityTree(lambda:self.data_x, 
                                 np.array(self.bounds, dtype=np.float), 
                                 lambda:self.data_c, 
                                 lambda:self.data_s,
                                 lambda:self.n_td_points() - self.time_window,
                                 self.self_weight,
                                 self.td_weight / float(self.time_window),
                                 lambda idx, c: self.set_data_c(idx, c), 
                                 self.max_points_per_region, 
                                 self.max_depth,
                                 self.split_mode, 
                                 self.competence_measure,
                                 self.progress_win_size, 
                                 self.progress_measure, 
                                 self.interest_measure, 
                                 self.sampling_mode,
                                 self.comp_min_cut,
                                 )
        
    def sample(self):
        if self.data_s is not None:
            return Tree.sample(self.tree)        
        else:
            return InterestTree.sample(self)
        
    def n_td_points(self): 
        if self.data_s is None:
            return 0
        else:
            return np.shape(self.data_s)[0]
    
    def add_top_down_goal(self, s):
        if self.data_s is None:
            self.data_s = np.array([s])
        else:
            self.data_s = np.append(self.data_s, np.array([s]), axis=0)      
        self.tree.add_top_down_goal(self.n_td_points() - 1)
        
        
        
interest_models = {'TDDensity': (TDDensityInterest, {'default': {'kernel': 'tophat',
                                                                 'bandwidth': 0.1,
                                                                 'time_window': 20,
                                                                 'self_weight': 0.5,
                                                                 'td_weight': 0.5}}),
                   'TDDensityTree': (TDDensityTreeInterest, {'default': {'time_window': 20,
                                                                         'self_weight': 0.5,
                                                                         'td_weight': 0.5}})
                   }
