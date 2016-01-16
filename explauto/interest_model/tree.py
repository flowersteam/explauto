import time
import numpy as np
import matplotlib.pyplot as plt

from heapq import heappop, heappush
from scipy.spatial.kdtree import minkowski_distance_p

from ..utils.utils import rand_bounds, softmax_choice
from ..utils.config import make_configuration
from .interest_model import InterestModel
from .competences import competence_exp, competence_dist



class InterestTree(InterestModel):
    """
    class InterestTree implements either R-IAC or SAGG-RIAC
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims, 
                 max_points_per_region, 
                 max_depth,
                 split_mode, 
                 competence_measure, 
                 progress_win_size, 
                 progress_measure,
                 interest_measure,
                 sampling_mode,
                 comp_min_cut):
        
        self.conf = conf
        self.bounds = self.conf.bounds[:, expl_dims]
        self.max_points_per_region = max_points_per_region
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.competence_measure = competence_measure
        self.progress_win_size = progress_win_size
        self.progress_measure = progress_measure
        self.interest_measure = interest_measure
        self.sampling_mode = sampling_mode
        self.comp_min_cut = comp_min_cut
        
        if progress_win_size >= max_points_per_region:
            raise ValueError("WARNING: progress_win_size should be < max_points_per_region")
        
        self.data_x = None
        self.data_c = None

        self.create_tree()
        
        InterestModel.__init__(self, expl_dims)
        
    def create_tree(self):
        self.tree = Tree(lambda:self.data_x, 
                         np.array(self.bounds, dtype=np.float), 
                         lambda:self.data_c,
                         lambda idx, c: self.set_data_c(idx, c), 
                         max_points_per_region=self.max_points_per_region, 
                         max_depth=self.max_depth,
                         split_mode=self.split_mode, 
                         competence_measure=self.competence_measure,
                         progress_win_size=self.progress_win_size, 
                         progress_measure=self.progress_measure, 
                         interest_measure=self.interest_measure,
                         sampling_mode=self.sampling_mode,
                         comp_min_cut=self.comp_min_cut)
        
    def add_x(self, x):
        if self.data_x is None:
            self.data_x = np.array([x])
        else:
            self.data_x = np.append(self.data_x, np.array([x]), axis=0)    
        
    def add_c(self, c):
        if self.data_c is None:
            self.data_c = np.array([c]) # Either prediction error or competence error
        else:
            self.data_c = np.append(self.data_c, c)             
            
    def set_data_c(self, idx, c):
        self.data_c[idx] = c        

    def sample(self):
        return self.tree.sample()
        
    def competence_pt(self, x):
        return self.tree.competence_leaf(self.tree.pt2leaf(x))
        
    def competence(self, mode="sw"):
        if self.data_x is None:
            return 0
        else:
            if mode == "sw":
                return self.tree.competence_idxs(range(self.n_points())[-self.tree.progress_win_size:])
            elif mode == "cum":
                return self.tree.competence_cum()
            else:
                raise NotImplementedError    
            
    def interest_pt(self, x):
        leaf = self.tree.pt2leaf(x)
        return self.tree.pt2leaf(x).progress
    
    def interest(self, mode="max"):
        if mode == "all":
            return self.tree.progress_all()
        elif mode == "max":
            return self.tree.max_interest
        else:
            raise NotImplementedError
                
    def n_points(self):
        if self.data_x is None:
            return 0
        else:
            return np.shape(self.data_x)[0]
        
    def update(self, xy, ms):
        self.add_x(xy[self.expl_dims])
        c = self.competence_measure(xy, ms)
        self.add_c(c)
        self.tree.add(self.n_points() - 1)


class Tree(object):
    """
        Interest Tree (recursive).
        
        This class also provides an index into a set of k-dimensional points which
        can be used to rapidly look up the nearest neighbors of any point.
    
        Parameters
        ----------
        get_data_x : (N,K) array
            Function that return the data points to be indexed.
        bounds_x : (2,K) array
            Bounds on tree's domain ([mins,maxs])
        get_data_c : (N,K) array_like
            Function that return the data points' competences.
        set_data_c : (N,K) array_like
            Function that set a data points' competence.
        max_points_per_region : int
            Maximum number of points per region. 
            A given region is splited when this number is exceeded.
        max_depth : int
            Maximum tree depth.
        split_mode : string
            Mode to split a region: random, median.
        competence_measure : func
            Function that gives the competence on the goal based on the reached point.
        progress_win_size : int
            Number of last points taken into account for progress computation 
            (should be < max_points_per_region)
        progress_measure : string
            How to compute progress: 'abs_deriv'
        interest_measure : list
            Weights to compute the interest: ['progress', 'volume'].
        sampling_mode : list 
            How to sample a point in the tree: ['greedy'], ['random'], 
            ['epsilon_greedy', eps], ['softmax', temperature] 
        comp_min_cut : float
            Minimal competence. If a competence is below that threshold, 
            it is reset (with set_data_c) to that value.
        idxs : list 
            List of indices to start with
        split_dim : int
            Dimension on which the next split will take place
        
        Raises
        ------
        RuntimeError
            The maximum recursion limit can be exceeded for large data
            sets.  If this happens, either increase the value for the `max_points_per_region`
            parameter or increase the recursion limit by::
    
                >>> import sys
                >>> sys.setrecursionlimit(10000)
    

    """
    def __init__(self, 
                 get_data_x, 
                 bounds_x, 
                 get_data_c, 
                 set_data_c, 
                 max_points_per_region=10, 
                 max_depth=10,
                 split_mode='median', 
                 competence_measure=competence_dist,
                 progress_win_size=5, 
                 progress_measure='abs_deriv', 
                 interest_measure=["progress"], 
                 sampling_mode=['softmax',0.1], 
                 comp_min_cut=True,
                 idxs=None, 
                 split_dim=0):
        
        self.get_data_x = get_data_x
        self.bounds_x = np.array(bounds_x, dtype=np.float64)
        self.get_data_c = get_data_c
        self.set_data_c = set_data_c
        self.max_points_per_region = max_points_per_region
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.competence_measure = competence_measure
        self.progress_win_size = progress_win_size
        self.progress_measure = progress_measure
        self.interest_measure = interest_measure
        self.sampling_mode = sampling_mode
        self.comp_min_cut = comp_min_cut

        self.split_dim = split_dim
        self.split_value = None
        self.lower = None
        self.greater = None
        self.idxs = idxs or []
        self.children = len(self.idxs)
        
        self.leafnode = True
        self.progress = 1.
        self.max_interest = 1.
        
        if self.children > self.max_points_per_region:
            self.split()
        self.compute_max_interest()
            
                    
    def get_leaves(self):
        """
        Get the list of all leaves.
        
        """
        return self.fold_up(lambda fl, fg: fl + fg, lambda leaf: [leaf])
        
    def depth(self):
        """
        Compute the depth of the tree (depth of a leaf=0).
        
        """
        return self.fold_up(lambda fl, fg: max(fl + 1, fg + 1), lambda leaf: 0)
        
    def volume(self):
        """
        Compute the volume of the node.
        
        """
        return np.prod(self.bounds_x[1,:] - self.bounds_x[0,:])
        
    def density(self):
        """
        Compute the density of the node.
        
        """
        return self.children / self.volume()    
    
    def pt2leaf(self, x):
        """
        Get the leaf which domain contains x.
        
        """
        if self.leafnode:
            return self
        else:
            if x[self.split_dim] < self.split_value:
                return self.lower.pt2leaf(x)
            else:
                return self.greater.pt2leaf(x)
                
    def sample_bounds(self):
        """
        Sample a point in the region of this node.
        
        """
        s = rand_bounds(self.bounds_x).flatten()
        return s
        
    def sample_random(self, rand_mode='by_volume'):
        """
        Sample Randomly .
        
        Parameters
        ----------
        rand_mode : string
            How to sample the tree: 'by_volume':
        
        """
        if rand_mode == 'by_leaf': 
            # Choose a leaf randomly
            return np.random.choice(self.get_leaves()).sample_bounds()
            
        elif rand_mode == 'by_volume': 
            # Choose a leaf weighted by volume, randomly
            if self.leafnode:
                return self.sample_bounds()
            else:
                split_ratio = ((self.split_value - self.bounds_x[0,self.split_dim]) / 
                               (self.bounds_x[1,self.split_dim] - self.bounds_x[0,self.split_dim]))
                if split_ratio > np.random.random():
                    return self.lower.sample(sampling_mode=['random'])
                else:
                    return self.greater.sample(sampling_mode=['random'])        
        
    def sample_greedy(self):
        """        
        Choose the leaf with the max interest.
        
        """    
        if self.leafnode:
            #print "Tree SAmple bounds", self.bounds_x
            return self.sample_bounds()
        else:
            lp = self.lower.max_interest
            gp = self.greater.max_interest
            if gp > lp:
                return self.greater.sample(sampling_mode=['greedy'])
            else:
                return self.lower.sample(sampling_mode=['greedy'])        
        
    def sample_epsilon_greedy(self, epsilon=0.1):
        """
        Choose the leaf with the max progress with probability (1-eps) 
        and a random leaf with probability (eps).
        
        Parameters
        ----------
        epsilon : float 
            
        """
        if epsilon > np.random.random():
            return self.sample(sampling_mode=['random'])
        else:
            return self.sample(sampling_mode=['greedy'])        
        
    def sample_softmax(self, temperature=1.):
        """
        Sample leaves with interest_measure weights plus a softmax exploration 
        (with a temperature parameter).
        
        Parameters
        ----------
        temperature : float 
        
        """
        if self.leafnode:
            return self.sample_bounds()
        else:
            leaves = self.get_leaves()
            interests = np.array(map(lambda leaf:leaf.max_interest, leaves))
            
            if np.isnan(np.sum(interests)): 
                # if progress_max = 0 or nan value in dataset, eps-greedy sample
                return self.sample_epsilon_greedy()
            else:
                leaf = leaves[softmax_choice(interests, temperature)]
                return leaf.sample_bounds()        
            
    def sample(self, sampling_mode=None):
        """
        Sample a point in the leaf region with max competence progress (recursive).
        
        Parameters
        ----------
        sampling_mode : list 
            How to sample a point in the tree: ['greedy'], ['random'], 
            ['epsilon_greedy', epsilon], ['softmax', temperature] 
            
        """
        if sampling_mode is None:
            sampling_mode = self.sampling_mode

        if sampling_mode[0] == 'random':
            return self.sample_random()
                
        elif sampling_mode[0] == 'greedy':
            return self.sample_greedy()
            
        elif sampling_mode[0] == 'epsilon_greedy':
            return self.sample_epsilon_greedy(sampling_mode[1])
            
        elif sampling_mode[0] == 'softmax':
            return self.sample_softmax(sampling_mode[1])
            
        else:
            raise NotImplementedError(sampling_mode)            
            
    def progress_all(self):
        """
        Competence progress on the last points of the overall tree.
        
        """
        if self.children > 4:
            return self.progress_idxs(range(self.children))
        else:
            return 1
                        
    def progress_idxs(self, idxs):
        """
        Competence progress on points of given indexes.
        
        """
        if self.progress_measure == 'abs_deriv_cov':
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                return abs(np.cov(zip(range(len(idxs)), self.get_data_c()[idxs]), rowvar=0)[0, 1])
            
        elif self.progress_measure == 'abs_deriv':
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                return np.abs(np.mean(np.diff(self.get_data_c()[idxs], axis=0)))
            
        elif self.progress_measure == 'abs_deriv_smooth':
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                v = self.get_data_c()[idxs]
                n = len(v)
                comp_beg = np.mean(v[:int(float(n)/2.)])
                comp_end = np.mean(v[int(float(n)/2.):])
                #print "Tree progress", idxs, v, n
                return np.abs(comp_end - comp_beg)
        else:
            raise NotImplementedError(self.progress_measure)        
        
    def competence_leaf(self, leaf):
        """
        Mean competence on points in given leaf
        """
        return self.competence_idxs(leaf.idxs)
    
    def competence_idxs(self, idxs):
        """
        Mean competence on points of given indexes.
        
        """
        comps = self.get_data_c()
        if comps is not None:
            return np.mean(self.get_data_c()[idxs])
        else:
            return 0.        
                
    def competence_cum(self): 
        """
        Compute mean competence from beginning on tree (recursive).
        
        """
        if self.leafnode:
            return self.competence_idxs(self.idxs)
        else:
            split_ratio = ((self.split_value - self.bounds_x[0,self.split_dim]) / 
                           (self.bounds_x[1,self.split_dim] - self.bounds_x[0,self.split_dim]))
            return self.lower.competence() * split_ratio + self.greater.competence() * (1 - split_ratio)        
    
    def compute_interest(self):
        """
        Compute the interest of a leaf depending on the weighting parameters.
        
        """
        self.max_interest = 1.
        if "progress" in self.interest_measure:
            self.max_interest = self.max_interest * self.progress
        if "volume" in self.interest_measure:
            self.max_interest = self.max_interest * self.volume()
            
    def recompute_tree_max_interest(self):
        """
        Recompute the max interest of all the nodes of the tree (recursive).
        
        """
        if self.leafnode:
            self.compute_max_interest()
        else:
            self.lower.recompute_tree_max_interest()
            self.greater.recompute_tree_max_interest()
            self.compute_max_interest()
        
    def compute_max_interest(self):
        """
        Compute max interest of sub-trees (not recursive).
        
        """
        if self.leafnode:
            self.progress = self.progress_idxs(self.idxs)
            self.compute_interest()
        else:
            self.max_interest = max(self.lower.max_interest, self.greater.max_interest)            
        
    def add(self, idx):
        """
        Add an index to the tree (recursive).
        
        """
        if self.leafnode and self.children >= self.max_points_per_region and self.max_depth > 0:
            self.split() 
        if self.leafnode:
            self.idxs.append(idx)
            leaf_add = self
            #process minimal competence
            if self.comp_min_cut:
                comp = self.get_data_c()[idx]
                comp_min = self.competence_measure(self.bounds_x[0,:], 
                                                   self.bounds_x[1,:])
                if comp < comp_min:
                    self.set_data_c(idx, comp_min)
        else:
            if self.get_data_x()[idx, self.split_dim] >= self.split_value:
                leaf_add = self.greater.add(idx)            
            else:
                leaf_add = self.lower.add(idx)
        self.compute_max_interest()
        self.children = self.children + 1
        return leaf_add # return leaf on which the point has been added    
    
    def split_random(self):
        """        
        Split randomly between min and max of node's points on split dimension.
        
        """
        split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
        split_min = min(split_dim_data)
        split_max = max(split_dim_data)
        return split_min + np.random.rand() * (split_max - split_min)
    
    def split_median(self):
        """
        Split on median (which fall on the middle of two points for even max_points_per_region) 
        of node's points on split dimension.
        
        """
        split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
        return np.median(split_dim_data)
        
    def split_middle(self):
        """
        Split on the middle of the region: might cause empty leaf.
        
        """
        split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
        return (self.bounds_x[0, self.split_dim] + self.bounds_x[1, self.split_dim]) / 2
        
    def split_best_interest_diff(self):
        """
        See Baranes2012: Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots
        choose between random split values the one that maximizes 'card(lower) * card(greater) * progress' 
        difference between the two.
        
        """
        split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
        split_min = min(split_dim_data)
        split_max = max(split_dim_data)
                    
        m = self.max_points_per_region # Constant that might be tuned: number of random split values to choose between
        rand_splits = split_min + np.random.rand(m) * (split_max - split_min)
        splits_fitness = np.zeros(m)
        for i in range(m):
            lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= rand_splits[i])[0]])
            greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > rand_splits[i])[0]])
            splits_fitness[i] = len(lower_idx) * len(greater_idx) * abs(self.progress_idxs(lower_idx) - 
                                                                       self.progress_idxs(greater_idx))
        return rand_splits[np.argmax(splits_fitness)]
        
    def choose_split_value(self):
        """
        Choose the split value depending on split mode.
        
        """
        if self.split_mode == 'random':
            return self.split_random()            
        elif self.split_mode == 'median':
            return self.split_median()            
        elif self.split_mode == 'middle':
            return self.split_middle()            
        elif self.split_mode == 'best_interest_diff':
            return self.split_best_interest_diff() 
        else:
            raise NotImplementedError
        
    def split(self):
        """
        Split the leaf node.
        
        """
        split_value = self.choose_split_value() 
        split_dim_data = self.get_data_x()[self.idxs, self.split_dim] # data on split dim
        lower_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= split_value)[0]])
        greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > split_value)[0]])

        self.leafnode = False
        self.idxs = None
        self.split_value = split_value
        
        split_dim = np.mod(self.split_dim + 1, np.shape(self.get_data_x())[1])
        
        l_bounds_x = np.array(self.bounds_x)
        l_bounds_x[1, self.split_dim] = split_value
        
        g_bounds_x = np.array(self.bounds_x)
        g_bounds_x[0, self.split_dim] = split_value
        
        self.create_subtrees(l_bounds_x, g_bounds_x, lower_idx, greater_idx, split_dim)
        
    def create_subtrees(self, l_bounds_x, g_bounds_x, lower_idx, greater_idx, split_dim):
        """
        Create lower and greater subtrees.
        
        """ 
        self.lower = Tree(self.get_data_x, 
                          l_bounds_x, 
                          self.get_data_c, 
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
        
        self.greater = Tree(self.get_data_x, 
                            g_bounds_x, 
                            self.get_data_c, 
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
            print "Leaf interest:", self.max_interest
            for _ in range(depth):
                print "    ",
            print "Leaf indices    :", self.idxs
            for _ in range(depth):
                print "    ",
            print "Leaf points     :", self.get_data_x()[self.idxs][:,0]
            for _ in range(depth):
                print "    ",
            print "Leaf competences:", self.get_data_c()[self.idxs]
        else:
            self.lower.print_tree(depth+1)
            self.greater.print_tree(depth+1)
                
    # Adapted from scipy.spatial.kdtree 
    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):

        side_distances = np.maximum(0,np.maximum(x-self.bounds_x[1],self.bounds_x[0]-x))
        if p != np.inf:
            side_distances **= p
            min_distance = np.sum(side_distances)
        else:
            min_distance = np.amax(side_distances)

        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the cell and the target
        #  distances between the nearest side of the cell and the target
        #  the head node of the cell
        q = [(min_distance,
              tuple(side_distances),
              self)]
        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance**p, i)
        neighbors = []

        if eps == 0:
            epsfac = 1
        elif p == np.inf:
            epsfac = 1/(1+eps)
        else:
            epsfac = 1/(1+eps)**p

        if p != np.inf and distance_upper_bound != np.inf:
            distance_upper_bound = distance_upper_bound**p

        while q:
            min_distance, side_distances, node = heappop(q)
            if node.leafnode:
                # brute-force
                data = self.get_data_x()[node.idxs]
                ds = minkowski_distance_p(data,x[np.newaxis,:],p)
                for i in range(len(ds)):
                    if ds[i] < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idxs[i]))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push cells that are too far onto the queue at all,
                # but since the distance_upper_bound decreases, we might get
                # here even if the cell's too far
                if min_distance > distance_upper_bound*epsfac:
                    # since this is the nearest cell, we're done, bail out
                    break
                # compute minimum distances to the children and push them on
                if x[node.split_dim] < node.split_value:
                    near, far = node.lower, node.greater
                else:
                    near, far = node.greater, node.lower

                # near child is at the same distance as the current node
                heappush(q,(min_distance, side_distances, near))

                # far child is further by an amount depending only
                # on the split value
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance, abs(node.split_value-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split_value-x[node.split_dim])
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
                else:
                    sd[node.split_dim] = np.abs(node.split_value-x[node.split_dim])**p
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

                # far child might be too far, if so, don't bother pushing it
                if min_distance <= distance_upper_bound*epsfac:
                    heappush(q,(min_distance, tuple(sd), far))

        if p == np.inf:
            return sorted([(-d,i) for (d,i) in neighbors])
        else:
            return sorted([((-d)**(1./p),i) for (d,i) in neighbors])
                
    # Adapted from scipy.spatial.kdtree 
    def nn(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """
        Query the tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : nonnegative float
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple if
            k is one, or tuple+(k,) if k is larger than one. Missing
            neighbors (e.g. when k > n or distance_upper_bound is
            given) are indicated with infinite distances.  If k is None,
            then d is an object array of shape tuple, containing lists
            of distances. In either case the hits are sorted by distance
            (nearest first).
        i : integer or array of integers
            The locations of the neighbors in self.data. i is the same
            shape as d.

        """
        self.n, self.m = np.shape(self.get_data_x())
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but has shape %s" % (self.m, np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k is None:
                dd = np.empty(retshape,dtype=np.object)
                ii = np.empty(retshape,dtype=np.object)
            elif k > 1:
                dd = np.empty(retshape+(k,),dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape+(k,),dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape,dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape,dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d,i) in hits]
                    ii[c] = [i for (d,i) in hits]
                elif k > 1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d,i) in hits], [i for (d,i) in hits]
            elif k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k,dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k,dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
                     
    def fold_up(self, f_inter, f_leaf):
        """
        Apply recursively the function f_inter from leaves to root, begining with function f_leaf on leaves.
        
        """
        return f_leaf(self) if self.leafnode else f_inter(self.lower.fold_up(f_inter, f_leaf), 
                                                          self.greater.fold_up(f_inter, f_leaf))                            
            
    def plot(self, ax, scatter=True, grid=True, progress_colors=True, progress_max=1., depth=10, plot_dims=[0,1]):
        """
        Plot a projection on 2D of the Tree.
        
        Parameters
        ----------
        ax : plt axis
        scatter : bool
            If the points are ploted
        grid : bool
            If the leaves' bounds are ploted
        progress_colors : bool
            If rectangles are filled with colors based on progress 
        progress_max : float
            Max progress on color scale (will be ploted as 1.)
        depth : int
            Max depth of the ploted nodes
        plot_dims : list
            List of the 2 dimensions to project tree on
        
        """
        
        if scatter and self.get_data_x() is not None:
            self.plot_scatter(ax, plot_dims)
            
        if grid:
            self.plot_grid(ax, progress_colors, progress_max, depth, plot_dims)
                
    def plot_scatter(self, ax, plot_dims=[0,1]):
        if np.shape(self.get_data_x())[0] <= 5000:
            ax.scatter(self.get_data_x()[:,plot_dims[0]], self.get_data_x()[:,plot_dims[1]], color = 'black')        
        
    def plot_grid(self, ax, progress_colors=True, progress_max=1., depth=10, plot_dims=[0,1]):
        if self.leafnode or depth == 0:
        
            mins = self.bounds_x[0,plot_dims]
            maxs = self.bounds_x[1,plot_dims]
            
            if progress_colors:
                prog_min = 0.
                c = plt.cm.jet((self.progress - prog_min) / (progress_max - prog_min)) if progress_max > prog_min else plt.cm.jet(0)
                ax.add_patch(plt.Rectangle(mins, maxs[0] - mins[0], maxs[1] - mins[1], color=c, alpha=0.7))
            else:
                ax.add_patch(plt.Rectangle(mins, maxs[0] - mins[0], maxs[1] - mins[1], fill=False))
                    
        else:
            self.lower.plot_grid(ax, progress_colors, progress_max, depth - 1, plot_dims)
            self.greater.plot_grid(ax, progress_colors, progress_max, depth - 1, plot_dims)

    


interest_models = {'tree': (InterestTree, {'default': {'max_points_per_region': 100,
                                                       'max_depth':15,
                                                       'split_mode': 'middle',
                                                       #'competence_measure': competence_dist,
                                                       'competence_measure': lambda target,reached : competence_exp(target, reached, 0.001, power=1.),
                                                       'progress_win_size': 20,
                                                       'progress_measure': 'abs_deriv_smooth',  
                                                       'interest_measure': ["progress", "volume"],              
                                                       'sampling_mode': ['softmax', 0.2],
                                                       'comp_min_cut':True}})}



if __name__ == '__main__': 
# Tested from explauto/ with python -m explauto.interest_model.tree


######################################
########## TEST TREE #################
######################################

    if True:
        print "\n########## TEST TREE #################"
        n = 100000
        k = 2
         
        bounds = np.zeros((2, k))
        bounds[1,:] = 1
     
        data_x = rand_bounds(bounds, n)
        data_c = np.random.rand(n, 1)
        def set_data_c(idx, c):
            data_c[idx] = c
        max_points_per_region = 5
        max_depth = 10
        split_mode = 'median'
        competence_measure = lambda target,reached : competence_exp(target, reached, 0.001, 1.)
        interest_measure = ["progress", "volume"]
        progress_win_size = 10
        sampling_mode = ['softmax', 0.1]
        comp_min_cut = True
        #print get_data_x, get_data_c

        tree = Tree(lambda:data_x, 
                    bounds, 
                    lambda:data_c, 
                    set_data_c,
                    max_points_per_region, 
                    max_depth,
                    split_mode, 
                    competence_measure,
                    progress_win_size, 
                    'abs_deriv_smooth', 
                    interest_measure, 
                    sampling_mode, 
                    comp_min_cut,
                    range(n))
          
        print "Sampling", tree.sample()
        print "Progress", tree.progress
        tree.add(42)
          
          
        ####### FIND Neighrest Neighbors (might be useful)
        t = time.time()
        dist, idx = tree.nn([0.5, 0.5], k=20)
        print "Time to find neighrest neighbors:", time.time() - t
        print data_x[idx]
         

######################################
########## TEST InterestTree #########
######################################

    if True:
        print "\n########## TEST InterestTree #########"
        
        np.random.seed(1)
        
        max_points_per_region = 20
        split_mode = 'best_interest_diff'
        #split_mode = 'median'
        #split_mode = 'middle'
        
        # WARNING: progress_win_size has to be < than max_points_per_region. 
        # If not, an improbably low competence will forever (in subtrees) 
        # be taken into account in the computation of progress, leading to high progress 
        # (if progress_measure='abs_deriv') and sampling forever in that region
        progress_win_size = 10  
        sampling_mode = ['softmax', 0.1]
        #sampling_mode = ['epsilon_greedy',0.2]
        #sampling_mode = ['greedy']
        max_depth = 10
        competence_measure = lambda target,reached : competence_exp(target, reached, 0.001, 1.)
        interest_measure = ["progress", "volume"]
        progress_win_size = 10
        comp_min_cut = True
    
        m_mins = [0, 0]
        m_maxs = [1, 1]
        s_mins = [3, 3]
        s_maxs = [4, 4]
        conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)
        
        expl_dims = [2, 3]
        
        riac = InterestTree(conf, 
                            expl_dims, 
                            max_points_per_region, 
                            max_depth,
                            split_mode, 
                            competence_measure,
                            progress_win_size, 
                            'abs_deriv_smooth', 
                            interest_measure, 
                            sampling_mode,
                            comp_min_cut)
        
        #print "Sample: ", riac.sample()
        
        # TEST UNIFORM RANDOM POINTS BATCH
        
        n = 1000
        xys = []
        mss = []
        for i in range(n):
            xys.append(rand_bounds(conf.bounds, 1)[0])
            mss.append(rand_bounds(conf.bounds, 1)[0])
              
        for i in range(n): # updated after for random seed purpose
            riac.update(xys[i], mss[i])
     
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        plt.xlim((riac.tree.bounds_x[0, 0], riac.tree.bounds_x[1, 0]))
        plt.ylim((riac.tree.bounds_x[0, 1], riac.tree.bounds_x[1, 1]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('R-IAC tiling')
        riac.tree.plot(ax, True, True, True, riac.interest())
          
        print "Max nb of children:", riac.tree.fold_up(lambda fl,fg:max(fl,fg), lambda leaf:leaf.children)
               
        print "Max leaf interest: ", riac.tree.max_interest
        import matplotlib.colorbar as cbar
        cax, _ = cbar.make_axes(ax) 
        cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet) 
        cb.set_label('Normalized Competence Progress')
              
        plt.show(block=False)
     

######################################
###### TEST PROGRESSING SAMPLING #####
######################################

         
    if True:
        print "\n###### TEST PROGRESSING SAMPLING #####"
        
        np.random.seed(1)
        
        max_points_per_region = 20
        split_mode = 'best_interest_diff'
        #split_mode = 'median'
        #split_mode = 'middle'
        
        # WARNING: progress_win_size has to be < than max_points_per_region. 
        # If not, an improbably low competence will forever (in subtrees) 
        # be taken into account in the computation of progress, leading to high progress 
        # (if progress_measure='abs_deriv') and sampling forever in that region
        progress_win_size = 10  
        sampling_mode = ['softmax',0.1]
        #sampling_mode = ['epsilon_greedy',0.2]
        #sampling_mode = ['greedy']
    
        m_mins = [0, 0]
        m_maxs = [1, 1]
        s_mins = [3, 3]
        s_maxs = [4, 4]
        conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)
        
        expl_dims = [2, 3]
        
        riac = InterestTree(conf, 
                            expl_dims, 
                            max_points_per_region, 
                            max_depth,
                            split_mode, 
                            competence_measure,
                            progress_win_size, 
                            'abs_deriv_smooth', 
                            interest_measure, 
                            sampling_mode,
                            comp_min_cut)
        
        n = 3000
             
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal')
        ax.set_xlim((riac.tree.bounds_x[0, 0], riac.tree.bounds_x[1, 0]))
        ax.set_ylim((riac.tree.bounds_x[0, 1], riac.tree.bounds_x[1, 1]))
        riac.tree.plot(ax, True, True, True, riac.interest())
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('R-IAC tiling')
        
        #print "Number of leaves:", len(riac.tree.get_leaves())
        #print "Max leaf progress: ", riac.tree.progress
        import matplotlib.colorbar as cbar
        cax, _ = cbar.make_axes(ax) 
        cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet) 
        cb.set_label('Normalized Competence Progress')
    #     cb2.set_ticks(np.linspace(0,riac.progress(), 5.))
    #     cb2.set_ticklabels(np.linspace(0,riac.progress(), 5.))
        plt.ion()
        plt.show()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    
        
        for i in range(n):
            # SAMPLE A POINT, COMPUTE A SIMULATED REACHED POINT, ADD IT TO THE INTEREST MODEL
            xy = rand_bounds(conf.bounds, 1)[0]
            ms = rand_bounds(conf.bounds, 1)[0]
            sample = riac.tree.sample()
            #print "Sampled point: ", sample
            #print np.random.choice(riac.tree.get_leaves()).sample_bounds()
            xy[expl_dims] = sample
            
            
            # HERE we try to simulate a competence based on the quantity of exploration in the region, 
            # with more progress in the middle of the map
            # Not sure how it can be interpreted
            # Need robotic setup for ecological testing
            
            leaf = riac.tree.pt2leaf(sample)
            density = leaf.density()
    #         if i > 0:
    #             print "comps", leaf.data_c[leaf.idxs]
            #print "Density:", density
            
            center = [3.5, 3.5]
            dist = np.linalg.norm(xy[expl_dims]- center)
            #print "dist", dist
            if density < 5000:
                dist_reached = 1. * 5000. / ((density + 1) * (dist + 0.1))
            else:
                dist_reached = 1. / (dist + 0.1)
                
            #ms[expl_dims] = np.random.normal(sample, eps)
            #print "dist_reached", dist_reached
            dist_reached = dist_reached + np.random.random() * 0.001
            ms[expl_dims] = [sample[0]+dist_reached, sample[1]]
            #print "Number of leaves:", len(riac.tree.get_leaves())
            #print sample
            
            
            # ADD SAMPLE
            riac.update(xy, ms)
            
            # UPDATE PLOT
            if np.mod(i + 1, 100) == 0:
                print "Iteration:", i + 1, " Tree depth:", riac.tree.depth(), " Progress:", riac.interest()
                ax.clear()
                riac.tree.plot(ax, False, True, True, 10., 12)#riac.progress())
                plt.draw()
                plt.show()
        
        ax.clear()
        riac.tree.plot(ax, True, True, True, riac.interest(), 12)
        ax.set_xlim((riac.tree.bounds_x[0, 0], riac.tree.bounds_x[1, 0]))
        ax.set_ylim((riac.tree.bounds_x[0, 1], riac.tree.bounds_x[1, 1]))
        plt.draw()
    
        print "Sampling in max progress region: ", riac.tree.sample(['greedy'])
        dists, idxs = riac.tree.nn(center, 10)
        print "Nearest Neighbors:", riac.data_x[idxs]
        
        plt.ioff()
    plt.show()
         