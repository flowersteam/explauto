from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.mixture import sample_gaussian
from numpy import array, vstack

from .. import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from ..third_party.models.models.dataset import DataSet

n_neighbors = 1
algorithm = 'kd_tree'


class NearestNeighbor(SensorimotorModel):

    def __init__(self, conf):
        self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        self.t = 0
        self.sigma_expl = (conf.m_maxs - conf.m_mins) / 10.
        self.data_m, self.data_s = [], []
        self.n_explore = 10
        self.to_explore = 0
        self.to_fit = True
    def infer(self, in_dims, out_dims, x):
        if self.t < n_neighbors:
            raise ExplautoBootstrapError
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            if self.to_fit:
                self.model = KNeighborsRegressor(n_neighbors=min(self.t, 10), weights='distance', algorithm=algorithm)
                self.to_fit = False
                self.model.fit(data_m)
            return self.predict(x)

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            self.

        else:
            raise NotImplementedError("NearestNeighbor only implements forward (M -> S)"
                                      "and inverse (S -> M) model, not general prediction")

    def update(self, m, s):
        if self.data_m:
            self.data_m = vstack((self.data_m, m))
        else:
            self.data_m = m
        if self.data_s:
            self.data_s = vstack((self.data_s, s))
        else:
            self.data_s = s
        self.to_fit = True
        self.t += 1
