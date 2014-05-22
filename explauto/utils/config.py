from numpy import hstack, vstack, digitize, linspace, array, dot, unravel_index, ravel_multi_index, product
from numpy.linalg import norm
from collections import namedtuple

from . import rand_bounds


Configuration = namedtuple('Configuration', ('m_mins', 'm_maxs', 's_mins', 's_maxs',
                                             'm_ndims', 's_ndims', 'ndims',
                                             'm_dims', 's_dims', 'dims',
                                             'm_bounds', 's_bounds', 'bounds'))


def make_configuration(m_mins, m_maxs, s_mins, s_maxs):
    m_ndims = len(m_mins)
    s_ndims = len(s_mins)
    ndims = m_ndims + s_ndims

    m_dims = range(m_ndims)
    s_dims = range(m_ndims, ndims)
    dims = m_dims + s_dims

    m_bounds = vstack((m_mins, m_maxs))
    s_bounds = vstack((s_mins, s_maxs))
    bounds = hstack((m_bounds, s_bounds))

    return Configuration(m_mins, m_maxs, s_mins, s_maxs,
                         m_ndims, s_ndims, ndims,
                         m_dims, s_dims, dims,
                         m_bounds, s_bounds, bounds)


class Space(object):
    def __init__(self, mins, maxs, cardinalities=None):
        self.mins, self.maxs = array(mins).astype(float), array(maxs).astype(float)
        self.ndims = self.mins.shape[0]
        self.dims = range(self.ndims)
        self.cardinalities = cardinalities
        self.widths = array(self.maxs) - array(self.mins)
        self.bin_widths = self.widths / self.cardinalities

        if cardinalities is not None:
            self.bins = [linspace(self.mins[d] + self.bin_widths[d], 
                         self.maxs[d] - self.bin_widths[d],
                         cardinalities[d] - 1) for d in range(self.ndims)]
            self.card = product(cardinalities)
        
    def discretize(self, values, dims):
        return array([digitize([values[d]], self.bins[d])[0] for d in dims])
        # return array([digitize(values[:, d], self.bins[d]) for d in dims]).T

    def index(self, values):
        # TODO use numpy.ravel_multi_index
        discrete = self.discretize(values, self.dims)
        # print discrete
        return self.multi2index(discrete)
        #return dot(discrete, hstack(([1.], array(self.cardinalities[:-1]))).reshape(-1, 1))

    def rand_value(self, index, n=1):
        multi_index = unravel_index(index, self.cardinalities)
        bins = [hstack((mins, bins, maxs)) for mins, bins, maxs in zip(self.mins, self.bins, self.maxs)]
        bounds = array([[b[i], b[i + 1]] for b, i in zip(bins, multi_index)]).T

        return rand_bounds(bounds, n)
                # vstack((self.mins, self.maxs)))

    def multi2index(self, multi_index):
        return ravel_multi_index(multi_index, self.cardinalities)

    def index2multi(self, index):
        return unravel_index(index, self.cardinalities)

    def test(self):
        ok = True
        for v in rand_bounds(vstack((self.mins, self.maxs)), n=100):
            # print v
            i = self.index(v)
            # print i
            v2 = self.rand_value(i)[0, :]
            # print v2
            i2 = self.index(v2)
            # print i2
            # print i == i2, norm(v - v2)
            ok &= i == i2
        return ok

    # def continuize(self, indexes):


