
try:
    import numpy as np
    import scipy.spatial
except:
    print("Can't import scipy.spatial (or numpy). Is scipy (or numpy) correctly installed ?")
    exit(1)

DATA_X = 0
DATA_Y = 1

class Databag(object):
    """Hold a set of vectors and provides nearest neighbors capabilities"""

    def __init__(self, dim):
        """
        :arg dim:  the dimension of the data vectors
        """
        self.dim = dim
        self.reset()

    def __repr__(self):
        return 'Databag(dim={0}, data=[{1}])'.format(self.dim, ', '.join(str(x) for x in self.data))

    def add(self, x):
        assert len(x) == self.dim
        self.data.append(np.array(x))
        self.size += 1
        self.nn_ready = False

    def reset(self):
        """Reset the dataset to zero elements."""
        self.data     = []
        self.size     = 0
        self.kdtree   = None  # KDTree
        self.nn_ready = False # if True, the tree is up-to-date.

    def nn(self, x, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the k nearest neighbors of x in the observed input data
        :arg x:      center
        :arg k:      the number of nearest neighbors to return (default: 1)
        :arg eps:    approximate nearest neighbors.
                     the k-th returned value is guaranteed to be no further than
                     (1 + eps) times the distance to the real k-th nearest neighbor.
        :arg p:      Which Minkowski p-norm to use. (default: 2, euclidean)
        :arg radius: the maximum radius (default: +inf)
        :return:     distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(np.array(x), k_x, radius = radius, eps = eps, p = p)

    def get(self, index):
        return self.data[index]

    def iter(self):
        return iter(self.data)

    def _nn(self, v, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Compute the k nearest neighbors of v in the observed data,
        :see: nn() for arguments descriptions.
        """
        self._build_tree()
        dists, idxes = self.kdtree.query(v, k = k, distance_upper_bound = radius,
                                         eps = eps, p = p)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self):
        """Build the KDTree for the observed data
        """
        if not self.nn_ready:
            self.kdtree   = scipy.spatial.cKDTree(self.data)
            self.nn_ready = True

    def __len__(self):
        return self.size


class Dataset(object):
    """Hold observations an provide nearest neighbors facilities"""

    @classmethod
    def from_data(cls, data):
        """ Create a dataset from an array of data, infering the dimension from the datapoint """
        if len(data) == 0:
            raise ValueError("data array is empty.")
        dim_x, dim_y = len(data[0][0]), len(data[0][1])
        dataset = cls(dim_x, dim_y)
        for x, y in data:
            assert len(x) == dim_x and len(y) == dim_y
            dataset.add_xy(x, y)
        return dataset

    @classmethod
    def from_xy(cls, x_array, y_array):
        """ Create a dataset from two arrays of data.

            :note: infering the dimensions for the first elements of each array.
        """
        if len(x_array) == 0:
            raise ValueError("data array is empty.")
        dim_x, dim_y = len(x_array[0]), len(y_array[0])
        dataset = cls(dim_x, dim_y)
        for x, y in zip(x_array, y_array):
            assert len(x) == dim_x and len(y) == dim_y
            dataset.add_xy(x, y)
        return dataset

    def __init__(self, dim_x, dim_y):
        """
            :arg dim_x:  the dimension of the input vectors
            :arg dim_y:  the dimension of the output vectors
        """
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.reset()

# The two next methods are used for plicling/unpickling the object (because cKDTree cannot be pickled).
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['kdtree']
        return odict

    def __setstate__(self,dict):
        self.__dict__.update(dict)
        self.nn_ready = [False, False]
        self.kdtree   = [None, None]


    def reset(self):
        """Reset the dataset to zero elements."""
        self.data     = [[], []]
        self.size     = 0
        self.kdtree   = [None, None]   # KDTreeX, KDTreeY
        self.nn_ready = [False, False] # if True, the tree is up-to-date.

    def add_xy(self, x, y):
        assert len(x) == self.dim_x and len(y) == self.dim_y
        self.data[0].append(np.append([1.0], x))
        self.data[1].append(np.array(y))
        self.size += 1
        self.nn_ready = [False, False]

    def get_x(self, index):
        return self.data[0][index][1:]

    def get_x_padded(self, index):
        return self.data[0][index]

    def get_y(self, index):
        return self.data[1][index]

    def get_xy(self, index):
        return self.get_x(index), self.get_y(index)

    def iter_x(self):
        return iter(d[1:] for d in self.data[0])

    def iter_y(self):
        return iter(self.data[1])

    def iter_xy(self):
        return zip(self.iter_x(), self.data[1])

    def __len__(self):
        return self.size

    def nn_x(self, x, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the k nearest neighbors of x in the observed input data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim_x
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(DATA_X, np.append([1.0], x), k = k_x, radius = radius, eps = eps, p = p)

    def nn_y(self, y, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the k nearest neighbors of y in the observed output data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(y) == self.dim_y
        k_y = min(k, self.size)
        return self._nn(DATA_Y, y, k = k_y, radius = radius, eps = eps, p = p)

    def _nn(self, side, v, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Compute the k nearest neighbors of v in the observed data,
        :arg side  if equal to DATA_X, search among input data.
                     if equal to DATA_Y, search among output data.
        @return  distance and indexes of found nearest neighbors.
        """
        self._build_tree(side)
        dists, idxes = self.kdtree[side].query(v, k = k, distance_upper_bound = radius,
                                               eps = eps, p = p)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self, side):
        """Build the KDTree for the observed data
        :arg side  if equal to DATA_X, build input data tree.
                     if equal to DATA_Y, build output data tree.
        """
        if not self.nn_ready[side]:
            self.kdtree[side]   = scipy.spatial.cKDTree(self.data[side])
            self.nn_ready[side] = True

class ScaledDataset(Dataset):
    """Hold observations an provide nearest neighbors facilities.
    Allow to assign and modify weights on input data dimensions.

    Due to some limitations, this will keep twice the data in memory.
    """

    @classmethod
    def from_data(cls, data, weights_x = None, weights_y = None):
        """Create a dataset from an array of data, infering the dimension from the datapoint"""

        if len(data) == 0:
            raise ValueError("data array is empty.")

        dim_x, dim_y = len(data[0][0]), len(data[0][1])
        weights_x = weights_x or np.array([1.0]*dim_x)
        weights_y = weights_y or np.array([1.0]*dim_y)

        dataset = cls(dim_x, dim_y, weights_x = weights_x, weights_y = weights_y)

        for dp in data:
            assert len(dp[0]) == dim_x and len(dp[1]) == dim_y
            dataset.add_xy(dp[0], dp[1])

        return dataset

    def __init__(self, dim_x, dim_y, weights_x = None, weights_y = None):
        """
        :arg dim_x:  the dimension of the input vectors
        :arg dim_y:  the dimension of the output vectors
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.weights_x = np.array([1.0]*(dim_x+1)) if weights_x is None else np.append([1.0], weights_x)
        self.weights_y = np.array([1.0]*dim_y) if weights_x is None else np.array(weights_y)

        self.reset()

    def reset(self):
        """Reset the dataset to zero elements."""
        Dataset.reset(self)
        self.wdata = [[], []]

    def add_xy(self, x, y):
        Dataset.add_xy(self, x, y)
        self.wdata[0].append(self.weights_x*self.data[0][-1])
        self.wdata[1].append(self.weights_y*self.data[1][-1])

    def nn_x(self, x, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the k nearest neighbors of x in the observed input data
        @see Databag.nn() for argument description
        :return:  distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim_x
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(DATA_X, self.weights_x*np.append([1.0], x), k = k_x, radius = radius, eps = eps, p = p)

    def nn_y(self, y, k = 1, radius = np.inf, eps = 0.0, p = 2):
        """Find the n nearest neighbors of y in the observed output data
        @see Databag.nn() for argument description
        :return:  distance and indexes of found nearest neighbors.
        """
        assert len(y) == self.dim_y
        k_y = min(k, self.size)
        return self._nn(DATA_Y, self.weights_y*y, k = k_y, radius = radius, eps = eps, p = p)

    def _build_tree(self, side):
        """Build the KDTree for the observed data
        :arg side:  if equal to DATA_X, build input data tree.
                    if equal to DATA_Y, build output data tree.
        """
        if not self.nn_ready[side]:
            self.kdtree[side]   = scipy.spatial.cKDTree(self.wdata[side])
            self.nn_ready[side] = True



# TODO: a logger plus propre
# try:
#     from cdataset import cDataset
#     pDataset = Dataset # keeping the python version accessible
#     Dataset = cDataset
# except ImportError:
#     import traceback
#     traceback.print_exc()
#     print("warning: cdataset.cDataset import error, defaulting to (slower) python implementation.")
