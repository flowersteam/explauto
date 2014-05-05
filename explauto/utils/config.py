from numpy import hstack, vstack
from collections import namedtuple


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
