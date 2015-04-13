from .interest_model import InterestModel
from .competences import competence_exp, competence_dist
from numpy import mean, median
from scipy.spatial.KDTree import minkowski_distance_p


class RiacInterest(InterestModel):
    def __init__(self, conf, expl_dims, max_points_per_region, split_mode, measure):
        
        self.conf = conf
        self.max_points_per_region = max_points_per_region
        self.bounds = self.conf.bounds[:, expl_dims]
        # self.ndims = bounds.shape[1]
        self.measure = measure #TODO check dist_min
        
        self.data_x = None
        self.data_c = None
        self.tree = Tree(data_x, data_c, max_points_per_region, split_mode)
        
        InterestModel.__init__(self, expl_dims)

    def sample(self):
        return rand_bounds(self.bounds).flatten()

    def update(self, xy, ms):
        # Add data, compute competence
        pass


interest_models = {'riac': (RiacInterest, {'default': {'max_points_per_region': 100,
                                                       'split_mode': 'median',
                                                       'measure': competence_dist}})}



# Based on scipy.spatial.KDTree
class Tree(object):
    """
    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points which
    can be used to rapidly look up the nearest neighbors of any point.

    Parameters
    ----------
    data : (N,K) array_like
        The data points to be indexed. This array is not copied, and
        so modifying this data will result in bogus results.
    max_points_per_region : int
        Maximum number of points per region. A given region is splited when this number is exceeded.

    Raises
    ------
    RuntimeError
        The maximum recursion limit can be exceeded for large data
        sets.  If this happens, either increase the value for the `leafsize`
        parameter or increase the recursion limit by::

            >>> import sys
            >>> sys.setrecursionlimit(10000)


    """
    def __init__(self, data_x, data_c, max_points_per_region, split_mode, idxs = [], split_dim = 0):
        #self.data = np.asarray(data)
        if data_x is not None:
            self.n, self.m = np.shape(self.data)
        self.data_x = data_x
        self.data_c = data_c
        self.max_points_per_region = max_points_per_region
        self.split_mode = split_mode

        self.split_dim = split_dim
        self.split = None
        self.less = None
        self.greater = None
        self.idxs = idxs
        self.children = len(self.idxs)
        
        self.leafnode = True
        if data_c is not None:
            self.competence = mean(self.data_c[self.idxs])#TODO get COMP
        else:
            self.competence = None
            
    
    def add(self, idx):
        """
        Add an index to the tree
        
        """
        if self.leafnode:
            if self.children >= self.max_points_per_region:
                self.split()
            else:                    
                self.idxs.append(idx)
                self.competence = mean(self.data_c[idxs])#TODO get COMP
        else:
            if data[idx] >= split:
                self.greater.add(idx)
            else:
                self.less.add(idx)
        self.children = self.children + 1
    
    
    def split(self):
        """
        Split the leaf node
        
        """
        if self.split_mode == 'median':
            split_dim_data = self.data_x[self.idxs,self.split_dim] # data on split dim
            split = median(split_dim_data)
            less_idx = np.nonzero(split_dim_data <= split)[0]
            greater_idx = np.nonzero(split_dim_data > split)[0]
            
            self.leafnode = False
            self.idxs = None
            self.split = split
            self.less = Tree(data, comp, max_points_per_region, less_idx)
            self.greater = Tree(data, comp, max_points_per_region, greater_idx)
        else:
            raise NotImplementedError
        

    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):

        side_distances = np.maximum(0,np.maximum(x-self.maxes,self.mins-x))
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
                data = self.data[node.idxs]
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
                if x[node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less

                # near child is at the same distance as the current node
                heappush(q,(min_distance, side_distances, near))

                # far child is further by an amount depending only
                # on the split value
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance, abs(node.split-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
                else:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])**p
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

                # far child might be too far, if so, don't bother pushing it
                if min_distance <= distance_upper_bound*epsfac:
                    heappush(q,(min_distance, tuple(sd), far))

        if p == np.inf:
            return sorted([(-d,i) for (d,i) in neighbors])
        else:
            return sorted([((-d)**(1./p),i) for (d,i) in neighbors])


    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
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
