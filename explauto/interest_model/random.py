from .interest_model import InterestModel
from ..utils import rand_bounds


class RandomInterest(InterestModel):
    def __init__(self, i_dims, bounds):
        InterestModel.__init__(self, i_dims)

        self.bounds = bounds[:, i_dims]
        # self.ndims = bounds.shape[1]

    def sample(self):
        return rand_bounds(self.bounds)

    def update(self, xy, ms):
        pass


interest_model = RandomInterest
configurations = {'default': {}}
