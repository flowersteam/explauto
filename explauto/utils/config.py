from numpy import hstack, vstack


class Configuration(object):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):
        self.m_mins = m_mins
        self.m_maxs = m_maxs
        self.s_mins = s_mins
        self.s_maxs = s_maxs

        self.m_ndims = len(self.m_mins)
        self.s_ndims = len(self.s_mins)
        self.ndims = self.m_ndims + self.s_ndims

        self.m_dims = range(self.m_ndims)
        self.s_dims = range(self.m_ndims, self.ndims)
        self.dims = self.m_dims + self.s_dims

        self.m_bounds = vstack((self.m_mins, self.m_maxs))
        self.s_bounds = vstack((self.s_mins, self.s_maxs))
        self.bounds = hstack((self.m_bounds, self.s_bounds))
