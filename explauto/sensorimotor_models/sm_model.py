
class SmModel(object):
    #def __init__(self, m_dims, s_dims):
        #self.m_dims = m_dims
        #self.s_dims = s_dims

    def infer(self, in_dims, out_dims):
        raise NotImplementedError

    def update(self, m, s):
        raise NotImplementedError

    def bootstrap(self, orders, stimuli):
        for i, m in enumerate(orders):
            self.update(m, stimuli[i,:])


