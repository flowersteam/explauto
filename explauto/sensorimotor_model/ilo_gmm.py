from numpy import array, hstack

from ..models.gmminf import GMM
from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from .dataset import Dataset

n_neighbors = 1


class IloGmm(SensorimotorModel):
    def __init__(self, conf, n_components=3):  # , n_components=None):
        SensorimotorModel.__init__(self, conf)
        # self.n_components = n_neighbors/20 if n_components is None else n_components
        self.n_components = n_components
        self.n_neighbors = max(100, (conf.ndims) ** 2)  # at least 100 neighbors
        self.n_neighbors = min(1000, self.n_neighbors)  # at most 1000 neighbors
        self.min_n_neighbors = 20  # otherwise raise ExplautoBootstrapError
        self.dataset = Dataset(conf.m_ndims, conf.s_ndims)
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        self.mode = ''

    def get_local_data(self, in_dims, out_dims, x):
        if self.dataset.size < self.min_n_neighbors:
            raise ExplautoBootstrapError
        if self.dataset.size < self.n_neighbors:
            n_neighbors = self.dataset.size
        else:
            n_neighbors = self.n_neighbors
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            dists, indexes = self.dataset.nn_x(x, k=n_neighbors)
            if self.mode == 'exploit':
                return self.dataset.get_y(indexes[0])

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            dists, indexes = self.dataset.nn_y(x, k=n_neighbors)
            if self.mode == 'exploit':
                return self.dataset.get_x(indexes[0])
        else:
            raise NotImplementedError("IloGmm only implements forward"
                                      "(M -> S) and inverse (S -> M) model, "
                                      "not general prediction")
        data = []
        for i in indexes:
            data.append(hstack(self.dataset.get_xy(i)))
        return array(data)

    def fit_local_gmm(self, in_dims, out_dims, x):
        gmm = GMM(n_components=self.n_components, covariance_type='full')
        gmm.fit(self.get_local_data(in_dims, out_dims, x))
        return gmm

    def compute_conditional_gmm(self, in_dims, out_dims, x):
        gmm = self.fit_local_gmm(in_dims, out_dims, x)
        return gmm.inference(in_dims, out_dims, x)

    def infer(self, in_dims, out_dims, x):
        gmm = self.compute_conditional_gmm(in_dims, out_dims, x)
        return gmm.sample().flatten()

    def update(self, m, s):
        self.dataset.add_xy(tuple(m), tuple(s))

configurations = {'default': {}}
sensorimotor_models = {'ilo_gmm': (IloGmm, configurations)}
