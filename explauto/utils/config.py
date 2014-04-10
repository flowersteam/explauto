from numpy import array, hstack, vstack

class Configuration(object):
    def __init__(self, config_dict):
        for k, v in config_dict.iteritems():
            setattr(self, k, v)
        self.m_ndims = len(self.m_mins)
        self.s_ndims = len(self.s_mins)
        self.ndims = self.m_ndims + self.s_ndims
        self.m_dims = range(self.m_ndims)
        self.s_dims = range(-self.s_ndims, 0)
        self.bounds = vstack((hstack((self.m_mins, self.s_mins)), hstack((self.m_maxs, self.s_maxs))))

    def __setattr__(self, name, value):
        """ "normalize" expl_dist and inf_dist in range(-s_ndims, m_ndims) """
        if name == "expl_dims" or name == "inf_dims":
            dims = array(value)
            dims[dims>=self.m_ndims] -= self.ndims
            super(Configuration, self).__setattr__(name, list(dims))
        else:
            super(Configuration, self).__setattr__(name, value)
        #self.expl_dims = array(self.expl_dims)
        #self.expl_dims[self.expl_dims>=self.m_ndims] -= self.ndims
    #def __setattr__(self, 'inf_dims'):
        #self.inf_dims = array(self.inf_dims)
        #self.inf_dims[self.inf_dims>=self.m_ndims] -= self.ndims
