from .interest_model import InterestModel
from ..utils import rand_bounds


class RandomInterest(InterestModel):
    def __init__(self, conf, expl_dims):
        InterestModel.__init__(self, expl_dims)

        self.bounds = conf.bounds[:, expl_dims]
        # self.ndims = bounds.shape[1]

    def sample(self):
        return rand_bounds(self.bounds)

    def update(self, xy, ms):
        pass


interest_models = {'random': (RandomInterest, {'default': {}})}
