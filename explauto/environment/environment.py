from numpy import array, minimum, maximum, zeros
from ..utils.config import Configuration 

class Environment(object):
    def __init__(self, **kwargs):
        conf = Configuration(kwargs)
        for k,v in conf.iteritems():
            setattr(self, k, v)
        #self.ndims = kwargs['ndims']
        #self.params = kwargs
        #self.n_sdims = config_dict['s_ndims']
        #self.bounds = array(config['bounds'])
        #self.default = array(config['default'])
        #self.inds_read = array(range(config['s_ndims']))
        #self.inds_write = array(range(config['m_ndims']))
        #self.ms_bounds = hstack((array(self.m_bounds), array(self.s_bounds)))
        self.state = zeros(self.ndims)

    def next_state(self, ag_state):
        raise NotImplementedError

    #def post_processing(self):
        #self.state = minimum(self.state, self.bounds[:,1])
        #self.state = maximum(self.state, self.bounds[:,0])

    def read(self):
        return self.state[self.readable]

    def write(self, data):
        self.state[self.writable] = data

    def dataset(self, orders):
        n = orders.shape[0]
        m_ndims = orders.shape[1]
        data = zeros((n, self.ndims))
        data[:, :m_ndims] = orders
        for i, m in enumerate(orders):
            self.next_state(m)
            data[i, m_ndims:] = self.state[m_ndims:]
        return data



