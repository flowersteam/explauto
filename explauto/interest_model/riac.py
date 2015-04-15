from .interest_model import InterestModel
from .competences import competence_exp, competence_dist
import numpy as np
from numpy import mean, median, linspace
from scipy.spatial.kdtree import minkowski_distance_p
from ..utils.utils import rand_bounds
from heapq import heappop, heappush

import matplotlib.pyplot as plt



class RiacInterest(InterestModel):
    def __init__(self, conf, expl_dims, max_points_per_region, split_mode, competence_measure, progress_win_size, progress_measure, sampling_mode):
        
        self.conf = conf
        self.bounds = self.conf.bounds[:, expl_dims]
        self.competence_measure = competence_measure
        
        self.data_x = None
        self.data_c = None

        self.tree = Tree(self.get_data_x, np.array(self.bounds, dtype=np.float), self.get_data_c, max_points_per_region, split_mode, progress_win_size, progress_measure, sampling_mode)
        
        InterestModel.__init__(self, expl_dims)

    def sample(self):
        return self.tree.sample()
    
    def progress(self):
        return self.tree.progress
    
    def get_data_x(self):
        return self.data_x
    
    def get_data_c(self):
        return self.data_c
    
    def update(self, xy, ms):
        #print np.shape(self.data_x), np.shape(np.array([xy[self.expl_dims]]))
        if self.data_x is None:
            self.data_x = np.array([xy[self.expl_dims]])
        else:
            self.data_x = np.append(self.data_x, np.array([xy[self.expl_dims]]), axis=0)
        if self.data_c is None:
            self.data_c = np.array([self.competence_measure(xy, ms, dist_min=0)])
        else:
            self.data_c = np.append(self.data_c, self.competence_measure(xy, ms, dist_min=0)) 
        self.tree.add(np.shape(self.data_x)[0]-1)





# Based on scipy.spatial.KDTree
class Tree(object):
    """
        Competence Progress Tree (recursive)
        
        This class provides an index into a set of k-dimensional points which
        can be used to rapidly look up the nearest neighbors of any point.
    
        Parameters
        ----------
        get_data_x : (N,K) array_like
            Function that return the data points to be indexed.
        bounds_x : 
            bounds on tree x domain
        get_data_c : (N,K) array_like
            Function that return the data points' competences.
        max_points_per_region : int
            Maximum number of points per region. A given region is splited when this number is exceeded.
        split_mode : string
            Mode to split a region: random, median.
        competence_measure : measure
        progress_win_size : Number of last points taken into account for progress computation
        progress_measure : how to compute progress: 'abs_deriv'
        
        Raises
        ------
        RuntimeError
            The maximum recursion limit can be exceeded for large data
            sets.  If this happens, either increase the value for the `max_points_per_region`
            parameter or increase the recursion limit by::
    
                >>> import sys
                >>> sys.setrecursionlimit(10000)
    

    """
    def __init__(self, get_data_x, bounds_x, get_data_c, max_points_per_region = 10, split_mode = 'median', progress_win_size = 5, progress_measure = 'abs_deriv', sampling_mode = 'best', idxs = [], split_dim = 0):
        
        self.get_data_x = get_data_x
        self.bounds_x = bounds_x
        self.get_data_c = get_data_c
        self.max_points_per_region = max_points_per_region
        self.split_mode = split_mode
        self.progress_win_size = progress_win_size
        self.progress_measure = progress_measure
        self.sampling_mode = sampling_mode

        self.split_dim = split_dim
        self.split_value = None
        self.less = None
        self.greater = None
        self.idxs = idxs
        self.children = len(self.idxs)
        
        self.leafnode = True
        self.progress = None
        
        #print self.idxs
        
        if self.children > self.max_points_per_region:
            self.split()
        self.compute_progress()
            
    def sample_bounds(self):
        s = rand_bounds(self.bounds_x).flatten()
        return s
        
    def get_leaves(self):
        return self.fold_up(lambda fl,fg:fl + fg, lambda leaf:[leaf])
    
    def depth(self):
        return self.fold_up(lambda fl,fg:max(fl+1,fg+1), lambda leaf:0)
    
    def volume(self):
        print "Volume", self.bounds_x, np.prod(self.bounds_x[1,:]-self.bounds_x[0,:])
        return np.prod(self.bounds_x[1,:]-self.bounds_x[0,:])
    
    def density(self):
        return self.children / self.volume()
    
    def pt2leaf(self, x):
        if self.leafnode:
            return self
        else:
            if x[self.split_dim] < self.split_value:
                return self.less.pt2leaf(x)
            else:
                return self.greater.pt2leaf(x)
        
        
    def sample(self, sampling_mode = None):
        """
        Sample a point in the leaf region with max competence progress (recursive)
        
        """
        if sampling_mode is None:
            sampling_mode = self.sampling_mode
            
        #print sampling_mode
        
        if sampling_mode == 'random':
            rand_mode = 'by_area'
            if rand_mode == 'by_leaf':
                return np.random.choice(self.get_leaves()).sample_bounds()
                
            elif rand_mode == 'by_area':
                if self.leafnode:
                    return self.sample_bounds()
                else:
                    split_ratio = (self.split_value-self.bounds_x[0,self.split_dim]) / (self.bounds_x[1,self.split_dim]-self.bounds_x[0,self.split_dim])
                    if split_ratio > np.random.random():
                        return self.less.sample(sampling_mode='random')
                    else:
                        return self.greater.sample(sampling_mode='random')
                
        elif sampling_mode == 'best':
            if self.leafnode:
                return self.sample_bounds()
            else:
                lp = self.less.progress
                gp = self.greater.progress
                if gp > lp:
                    return self.greater.sample(sampling_mode='best')
                else:
                    return self.less.sample(sampling_mode='best')
                
        elif sampling_mode == 'epsilon_gready':
            eps = 0.2
            if eps > np.random.random():
                return self.sample(sampling_mode='random')
            else:
                return self.sample(sampling_mode='best')
                
        elif sampling_mode == 'softmax':
            pass
        else:
            raise NotImplementedError
            
            
    def progress_all(self):
        """
        Competence progress of the overall tree
        
        """
        return self.progress_idxs(range(np.shape(self.get_data_x())[0] - self.progress_win_size, np.shape(self.get_data_x())[0]))
    
            
    def progress_idxs(self, idxs):
        """
        Competence progress on points of given indexes
        
        """
        if self.progress_measure == 'abs_deriv':
            if len(idxs) <= 1:
                return 0
            else:
                idxs = sorted(idxs)[- self.progress_win_size:]
                return abs(np.cov(zip(range(len(idxs)), self.get_data_c()[idxs]), rowvar=0)[0, 1])
        else:
            raise NotImplementedError
        
    
    def compute_progress(self):
        """
        Compute max competence progress of sub-trees (not recursive)
        
        """
        if self.leafnode:
            self.progress = self.progress_idxs(self.idxs)
        else:
            self.progress = max(self.less.progress, self.greater.progress)
            
        
    def add(self, idx):
        """
        Add an index to the tree (recursive)
        
        """
        if self.leafnode and self.children >= self.max_points_per_region:
            self.split() 
        if self.leafnode:
            self.idxs.append(idx)
            leaf_add = self
        else:
            if self.get_data_x()[idx, self.split_dim] >= self.split_value:
                leaf_add = self.greater.add(idx)            
            else:
                leaf_add = self.less.add(idx)
        self.compute_progress()
        self.children = self.children + 1
        return leaf_add # return leaf on which the point has been added
    
    
    def split(self):
        """
        Split the leaf node
        
        """
        if self.split_mode == 'random':
            split_dim_data = self.get_data_x()[self.idxs,self.split_dim] # data on split dim
            split_min = min(split_dim_data)
            split_max = max(split_dim_data)
            split_value = split_min + np.random.rand() * (split_max - split_min)
            
        elif self.split_mode == 'median':
            split_dim_data = self.get_data_x()[self.idxs,self.split_dim] # data on split dim
            split_value = median(split_dim_data)
            #print "median split", split_dim_data, split_value
            
        elif self.split_mode == 'middle':
            split_dim_data = self.get_data_x()[self.idxs,self.split_dim] # data on split dim
            split_value = (self.bounds_x[0, self.split_dim] + self.bounds_x[1, self.split_dim])/2
            
        # See Baranes2012: Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots
        elif self.split_mode == 'best_interest_diff': 
            split_dim_data = self.get_data_x()[self.idxs,self.split_dim] # data on split dim
            split_min = min(split_dim_data)
            split_max = max(split_dim_data)
                        
            m = self.max_points_per_region
            rand_splits = split_min + np.random.rand(m) * (split_max - split_min)
            splits_fitness = np.zeros(m)
            for i in range(m):
                less_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= rand_splits[i])[0]])
                greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > rand_splits[i])[0]])
                splits_fitness[i] = len(less_idx) * len(greater_idx) * abs(self.progress_idxs(less_idx) - self.progress_idxs(greater_idx))
            split_value = rand_splits[np.argmax(splits_fitness)]
            #print "best_interest_diff: ", rand_splits, splits_fitness, split_value
            
        else:
            raise NotImplementedError
    
        less_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data <= split_value)[0]])
        greater_idx = list(np.array(self.idxs)[np.nonzero(split_dim_data > split_value)[0]])
        #print "Split idxs:", self.idxs, less_idx, greater_idx
        self.leafnode = False
        self.idxs = None
        self.split_value = split_value
        
        split_dim = np.mod(self.split_dim + 1, np.shape(self.get_data_x())[1])
        
        l_bounds_x = np.array(self.bounds_x)
        l_bounds_x[1, self.split_dim] = split_value
        
        g_bounds_x = np.array(self.bounds_x)
        g_bounds_x[0, self.split_dim] = split_value
        
        #print "Split value", split_value#, l_bounds_x, g_bounds_x
        self.less = Tree(self.get_data_x, l_bounds_x, self.get_data_c, self.max_points_per_region, self.split_mode, self.progress_win_size, self.progress_measure, self.sampling_mode, idxs = less_idx, split_dim = split_dim)
        self.greater = Tree(self.get_data_x, g_bounds_x, self.get_data_c, self.max_points_per_region, self.split_mode, self.progress_win_size, self.progress_measure, self.sampling_mode, idxs = greater_idx, split_dim = split_dim)
        

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
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less

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
            
            
    def map_node(self, f, mode='before'):
        if self.leafnode:
            f(self)
        else:
            if mode == 'before':
                f(self)
            self.less.map_node(f)
            if mode == 'middle':
                f(self)
            self.greater.map_node(f)
            if mode == 'after':
                f(self)
                
                
    def fold_up(self, f_inter, f_leaf):
        if self.leafnode:
            return f_leaf(self)
        else:
            return f_inter(self.less.fold_up(f_inter, f_leaf), 
                           self.greater.fold_up(f_inter, f_leaf))
                        
                
    def map_inter_node(self, f, mode='before'):
        if not self.leafnode:
            if mode == 'before':
                f(self)
            self.less.map_inter_node(f)
            if mode == 'middle':
                f(self)
            self.greater.map_inter_node(f)
            if mode == 'after':
                f(self)
            
            
    def map_leaf_node(self, f):
        if not self.leafnode:
            self.less.map_leaf_node(f)
            self.greater.map_leaf_node(f)
        else:
            f(self)
            
                
        
            
    def plot(self, ax, scatter = True, grid = True, progress_colors = True, progress_max = 1., depth = 10):
        
        if self.leafnode or depth == 0:
            
            if grid:
                mins = self.bounds_x[0]
                maxs = self.bounds_x[1]
                
                if progress_colors:
                    prog_min = 0.
                    if progress_max > prog_min:
                        c = plt.cm.jet((self.progress - prog_min) / (progress_max - prog_min))
                    else:
                        c = plt.cm.jet(0)
                    #print (self.progress - prog_min) / (progress_max - prog_min)
                    
                    ax.add_patch(plt.Rectangle(mins, maxs[0]-mins[0], maxs[1]-mins[1], color=c, alpha = 0.7))
                    
                else:
                    ax.add_patch(plt.Rectangle(mins, maxs[0]-mins[0], maxs[1]-mins[1], fill=False))
                    
        else:
            self.less.plot(ax, False, grid, progress_colors, progress_max, depth-1)
            self.greater.plot(ax, False, grid, progress_colors, progress_max, depth-1)
        
        
        if scatter and self.get_data_x() is not None and np.shape(self.get_data_x())[0] <= 5000:
            ax.scatter(self.get_data_x()[:,0], self.get_data_x()[:,1], color = 'black')








interest_models = {'riac': (RiacInterest, {'default': {'max_points_per_region': 100,
                                                       'split_mode': 'best_interest_diff',
                                                       'competence_measure': competence_dist,
                                                       'progress_win_size': 10,
                                                       'progress_measure': 'abs_deriv',                                                       
                                                       'sampling_mode': 'best'}})}








if __name__ == '__main__': 
    
#     n = 100000
#     k = 2
#     
#     bounds = np.zeros((2,k))
#     bounds[1,:] = 1
# 
#     data_x = rand_bounds(bounds, n)
#     data_c = np.random.rand(n,1)
#     
#     def get_data_x():
#         return data_x
#     
#     def get_data_c():
#         return data_c
#     
#     max_points_per_region = 5
#     split_mode = 'median'
#     progress_win_size = 10
#     
#     print get_data_x, get_data_c
#      
#     tree = Tree(get_data_x, bounds, get_data_c, max_points_per_region, split_mode, competence_dist, progress_win_size, 'abs_deriv', range(n))
#      
#     print tree.sample()
#     print tree.progress
#     tree.add(42)
#      
#      
#     ####### FIND Neighrest Neighbors (might be useful)
#     import time
#     t = time.time()
#     dist, idx = tree.nn([0.5,0.5], k=20)
#     print "Time to find neighrest neighbors:", time.time() - t
#     print data_x[idx]
#     
    ####### TEST RiacInterest
    from ..utils.config import make_configuration
    from ..utils.utils import rand_bounds
    
    np.random.seed(1)
    
    max_points_per_region = 20
    #split_mode = 'best_interest_diff'
    split_mode = 'median'
    progress_win_size = 20
    sampling_mode = 'epsilon_gready'

    m_mins = [0,0]
    m_maxs = [1,1]
    s_mins = [3,3]
    s_maxs = [4,4]
    conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)
    
    expl_dims = [2,3]
    
    riac = RiacInterest(conf, expl_dims, max_points_per_region, split_mode, competence_dist, progress_win_size, 'abs_deriv', sampling_mode)
    
    print "Sample: ", riac.sample()
    
    
    # TEST UNIFORM RANDOM POINTS
    
#     n = 1000
#     xys = []
#     mss = []
#     for i in range(n):
#         xys.append(rand_bounds(conf.bounds, 1)[0])
#         mss.append(rand_bounds(conf.bounds, 1)[0])
#         
#     for i in range(n): # updated after for random seed purpose
#         riac.update(xys[i], mss[i])

#     fig1 = plt.figure()
#     ax = fig1.add_subplot(111, aspect='equal')
#     plt.xlim((riac.tree.bounds_x[0,0],riac.tree.bounds_x[1,0]))
#     plt.ylim((riac.tree.bounds_x[0,1],riac.tree.bounds_x[1,1]))
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('R-IAC tiling')
#     riac.tree.plot(ax, True, True, True, riac.progress())
#     
#          
#     print "Max leaf progress: ", riac.tree.progress
#     import matplotlib.colorbar as cbar
#     cax, _ = cbar.make_axes(ax) 
#     cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet) 
#     cb.set_label('Normalized Competence Progress')
         
         

# TEST PROGRESSING SAMPLING
         
    n = 20000
         
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, aspect='equal')
    plt.xlim((riac.tree.bounds_x[0,0],riac.tree.bounds_x[1,0]))
    plt.ylim((riac.tree.bounds_x[0,1],riac.tree.bounds_x[1,1]))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('R-IAC tiling')
    riac.tree.plot(ax, True, True, True, riac.progress())
    
    print "Number of leaves:", len(riac.tree.get_leaves())
    print "Max leaf progress: ", riac.tree.progress
    import matplotlib.colorbar as cbar
    cax, _ = cbar.make_axes(ax) 
    cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet) 
    cb.set_label('Normalized Competence Progress')
#     cb2.set_ticks(linspace(0,riac.progress(), 5.))
#     cb2.set_ticklabels(linspace(0,riac.progress(), 5.))
    plt.ion()
    
    
    for i in range(n):
        xy = rand_bounds(conf.bounds, 1)[0]
        ms = rand_bounds(conf.bounds, 1)[0]
        sample = riac.tree.sample()
        print "Sampled point: ", sample
        #print np.random.choice(riac.tree.get_leaves()).sample_bounds()
        xy[expl_dims] = sample
        
        leaf = riac.tree.pt2leaf(sample)
        density = leaf.density()
        print "Density:", density
        
        center = [3.5, 3.5]
        dist = np.linalg.norm(xy[expl_dims]- center)
        dist_reached = 0.1*100./(density*dist) + 0.001
        #ms[expl_dims] = np.random.normal(sample, eps)
        ms[expl_dims] = [sample[0]+dist_reached, sample[1]]

        #print "Number of leaves:", len(riac.tree.get_leaves())
        #print sample
        
        
        
        riac.update(xy, ms)
        if np.mod(i+1,100)==0:
            print i+1
            print "Tree depth:", riac.tree.depth()
            print "Progress", riac.progress()
            ax.clear()
            riac.tree.plot(ax, False, True, True, 1., 12)#riac.progress())
            plt.draw()
            plt.show()
    
    plt.ioff()
    plt.show()
     