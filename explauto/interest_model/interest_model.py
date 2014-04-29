class InterestModel(object):
    def __init__(self, i_dims):
        self.i_dims = i_dims

    def sample(self):
        raise NotImplementedError

    def update(self, xy, ms):
        raise NotImplementedError
