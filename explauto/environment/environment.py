from numpy import array, minimum, maximum, zeros

class Environment(object):
    def __init__(self, ndims):
        self.ndims = ndims
        #self.n_sdims = config_dict['s_ndims']
        #self.bounds = array(config['bounds'])
        #self.default = array(config['default'])
        #self.inds_read = array(range(config['s_ndims']))
        #self.inds_write = array(range(config['m_ndims']))
        #self.ms_bounds = hstack((array(self.m_bounds), array(self.s_bounds)))
        self.state = zeros(ndims)

    def next_state(self, ag_state):
        raise NotImplementedError

    #def post_processing(self):
        #self.state = minimum(self.state, self.bounds[:,1])
        #self.state = maximum(self.state, self.bounds[:,0])

    def read(self):
        return self.state[self.inds_out]

    def write(self, data):
        self.state[self.inds_in] = data

