from numpy import array, hstack

from ..models.gmminf import GMM
from .. import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from ..third_party.models.models.dataset import Dataset

n_neighbors = 1


class IloGmm(SensorimotorModel):
    def __init__(self, conf, n_neighbors=100, n_components=None):
        self.n_components = n_neighbors/20 if n_components is None else n_components

        self.dataset = Dataset(conf.m_ndims, conf.s_ndims)
        self.n_neighbors = n_neighbors
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims
        self.mode = ''

    def infer(self, in_dims, out_dims, x):
        if self.dataset.size < self.n_neighbors:
            raise ExplautoBootstrapError
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            dists, indexes = self.dataset.nn_x(x, k=self.n_neighbors)
            if self.mode == 'exploit':
                return self.dataset.get_y(indexes[0])

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            dists, indexes = self.dataset.nn_y(x, k=self.n_neighbors)
            if self.mode == 'exploit':
                return self.dataset.get_x(indexes[0])
        else:
            raise NotImplementedError("IloGmm only implements forward"
                                      "(M -> S) and inverse (S -> M) model, "
                                      "not general prediction")
        data = []
        for i in indexes:
            data.append(hstack(self.dataset.get_xy(i)))
        data = array(data)
        gmm = GMM(n_components=self.n_components, covariance_type='full')
        gmm.fit(data)
        return gmm.inference(in_dims, out_dims, x).sample()

    def update(self, m, s):
        self.dataset.add_xy(tuple(m), tuple(s))

configurations = {'default': {}}
sensorimotor_models = {'ilo_gmm': (IloGmm, configurations)}
