import numpy

from sklearn.preprocessing import StandardScaler

from ..utils import rand_bounds
from ..models.gmminf import GMM
from .competences import competence_exp
from .interest_model import InterestModel


class GmmInterest(InterestModel):
    def __init__(self, conf, expl_dims, measure, n_samples=40, n_components=6):
        InterestModel.__init__(self, expl_dims)

        self.measure = measure
        self.bounds = conf.bounds[:, expl_dims]
        self.n_components = n_components
        self.scale_t = 1  # 1. / n_samples
        self.t = - self.scale_t * n_samples
        self.scale_x = conf.bounds[1, expl_dims] - conf.bounds[0, expl_dims]
        self.scale_measure = abs(measure(numpy.zeros_like(conf.bounds[0, :]),
                                         numpy.zeros_like(conf.bounds[0])))

        self.data = numpy.zeros((n_samples, len(expl_dims) + 2))
        self.n_samples = n_samples
        self.scaler = StandardScaler()

        for _ in range(n_samples):
            self.update(rand_bounds(conf.bounds), rand_bounds(conf.bounds))

    def sample(self):
        x = self.gmm_choice.sample()
        x = self.scaler.inverse_transform(numpy.hstack(([0.], x.flatten(), [0.])))[1:-1]
        x = numpy.maximum(x, self.bounds[0, :])
        x = numpy.minimum(x, self.bounds[1, :])
        return x.T

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        self.data[self.t % self.n_samples, 0] = self.t
        self.data[self.t % self.n_samples, -1] = measure
        self.data[self.t % self.n_samples, 1:-1] = xy.flatten()[self.expl_dims]

        self.t += self.scale_t
        if abs(self.t % (self.n_samples * self.scale_t / 4.)) < self.scale_t:
            self.update_gmm()

        return self.t, xy.flatten()[self.expl_dims], measure

    def update_gmm(self):
        scaled_data = self.scaler.fit_transform(self.data)

        self.gmm = GMM(n_components=self.n_components, covariance_type='full')
        self.gmm.fit(numpy.array(scaled_data))
        self.gmm_choice = self.gmm_interest()

    def gmm_interest(self):
        cov_t_c = numpy.array([self.gmm.covars_[k, 0, -1]
                               for k in range(self.gmm.n_components)])
        cov_t_c = numpy.exp(cov_t_c)
        # cov_t_c[cov_t_c <= 1e-100] = 1e-100

        gmm_choice = self.gmm.inference([0], range(1, len(self.expl_dims) + 1), [1.])
        gmm_choice.weights_ = cov_t_c
        gmm_choice.weights_ /= numpy.array(gmm_choice.weights_).sum()

        return gmm_choice

interest_models = {'gmm_progress_beta': (GmmInterest,
                                         {'default': {'measure': competence_exp,
                                                      'n_samples': 40,
                                                      'n_components': 6}})}
