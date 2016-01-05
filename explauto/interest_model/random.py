from .interest_model import InterestModel
from ..utils import rand_bounds


class RandomInterest(InterestModel):
    def __init__(self, conf, expl_dims):
        InterestModel.__init__(self, expl_dims)

        self.bounds = conf.bounds[:, expl_dims]
        self.ndims = self.bounds.shape[1]

    def sample(self):
        return rand_bounds(self.bounds).flatten()

    def update(self, xy, ms):
        pass
    
    def sample_given_context(self, c, c_dims):
        '''
        Sample randomly on dimensions not in context
            c: context value on c_dims dimensions, not used
            c_dims: w.r.t sensori space dimensions
        '''
        return self.sample()[list(set(range(self.ndims)) - set(c_dims))]


interest_models = {'random': (RandomInterest, {'default': {}})}
