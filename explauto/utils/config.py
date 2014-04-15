from numpy import array, hstack, vstack

class Configuration(dict):
    def __init__(self, config_dict):
        dict.__init__(self, config_dict)
        #for k, v in config_dict.iteritems():
            #setattr(self, k, v)
        self['m_ndims'] = len(self.m_mins)
        self['s_ndims'] = len(self.s_mins)
        self['ndims'] = self.m_ndims + self.s_ndims
        self['m_dims'] = range(self.m_ndims)
        self['s_dims'] = range(-self.s_ndims, 0)
        self['m_bounds'] = vstack((self.m_mins, self.m_maxs)) 
        self['s_bounds'] = vstack((self.s_mins, self.s_maxs))
        self['bounds'] = hstack((self.m_bounds, self.s_bounds))

    def __getattr__(self, attr):
        return self[attr]    

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, name, value):
        """ "normalize" expl_dims and inf_dims in range(-s_ndims, m_ndims) """
        if name == "expl_dims" or name == "inf_dims":
            value = array(value)
            value[value>=self.m_ndims] -= self.ndims
            value = list(value)
        super(Configuration, self).__setitem__(name, value)
