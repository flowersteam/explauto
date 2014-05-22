from sklearn.mixture import sample_gaussian
from numpy import inf, ones  # array, vstack
from numpy.linalg import norm

from .. import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from ..third_party.models.models.dataset import Dataset

n_neighbors = 1
# algorithm = 'kd_tree'


class NearestNeighbor(SensorimotorModel):

    def __init__(self, conf, sigma_ratio):
        """ This class implements a simple sensorimotor model inspired from the original SAGG-RIAC algorithm. Used as a forward model, it simply returns the image in S of nearest neighbor in M. Used as an inverse model, it looks at the nearest neighbor in S, then explore during the n_explore following calls to self.infer around that neighborhood.

        :param conf: a Configuration instance
        :type conf: :class:`~explauto.utils.config.Configuration`

        :param int n_explore: the number of exploration trials for each goal
        :param float sigma_ration: the standard deviation of the exploration will be (conf.m_maxs - conf.m_mins) sigma_ratio. Hence, low values of this parameters means larger exploration
        """
        self.dataset = Dataset(conf.m_ndims, conf.s_ndims)
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        self.t = 0
        self.sigma_expl = (conf.m_maxs - conf.m_mins) * float(sigma_ratio)
        self.n_explore = 1
        self.to_explore = 0
        self.best_dist_to_goal = float('inf')
        self.current_goal = inf * ones(conf.s_ndims)
        self.mode = 'explore'

    def infer(self, in_dims, out_dims, x):
        if self.t < n_neighbors:
            raise ExplautoBootstrapError
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            dists, indexes = self.dataset.nn_x(x, k=1)
            return self.dataset.get_y(indexes[0])

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if self.mode == 'explore':
                if not self.to_explore:
                    self.current_goal = x
                    dists, indexes = self.dataset.nn_y(x, k=1)
                    self.mean_explore = self.dataset.get_x(indexes[0])
                    self.to_explore = self.n_explore
                self.to_explore -= 1
                return sample_gaussian(self.mean_explore, self.sigma_expl ** 2)
            else:  # exploit'
                dists, indexes = self.dataset.nn_y(x, k=1)
                return self.dataset.get_x(indexes[0])

        else:
            raise NotImplementedError("NearestNeighbor only implements forward"
                                      "(M -> S) and inverse (S -> M) model, "
                                      "not general prediction")

    def update(self, m, s):
        if self.t > 0:
            dist_to_goal = norm(self.current_goal - s)
            if dist_to_goal < self.best_dist_to_goal:
                self.mean_explore = m
                self.best_dist_to_goal = dist_to_goal

        self.dataset.add_xy(tuple(m), tuple(s))
        self.t += 1

configurations = {'default': {'sigma_ratio': 1./6.}}
sensorimotor_models = {'nearest_neighbor': (NearestNeighbor, configurations)}
