import numpy

from sklearn.preprocessing import StandardScaler

from ..utils import rand_bounds
from ..utils import bounds_min_max
from ..utils import is_within_bounds
from ..models.gmminf import GMM
from .competences import competence_exp
from .interest_model import InterestModel


class GmmInterest(InterestModel):
    def __init__(self, conf, expl_dims, measure, n_samples=40, n_components=6, update_frequency=10, resample_if_out_of_bounds=False):
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
        self.update_frequency = update_frequency
        self.resample_if_out_of_bounds = resample_if_out_of_bounds

        for _ in range(n_samples):
            self.update(rand_bounds(conf.bounds), rand_bounds(conf.bounds))

    def sample(self):
        out_of_bounds = True
        while out_of_bounds:
            # sample from gmm
            x = self.gmm_choice.sample()
            # inverse the transform applied on data
            x = self.scaler.inverse_transform(numpy.hstack(([0.], x.flatten(), [0.])))[1:-1]
            # if resample_if_out_of_bounds is True, we check if x is in bounds and resample while not within bounds
            if self.resample_if_out_of_bounds:
                # if within bound let go out of the while loop
                if is_within_bounds(x, self.bounds[0, :], self.bounds[1, :]):
                    out_of_bounds = False
                #else we keep out_of_bounds to True
            else:
                # just bound the result and let go out of while loop
                x = bounds_min_max(x, self.bounds[0, :], self.bounds[1, :])
                out_of_bounds = False
        return x

    def update(self, xy, ms):
        measure = self.measure(xy, ms)
        self.data[self.t % self.n_samples, 0] = self.t
        self.data[self.t % self.n_samples, -1] = measure
        self.data[self.t % self.n_samples, 1:-1] = xy.flatten()[self.expl_dims]

        self.t += self.scale_t
        if self.t >= 0:
            if self.t % self.update_frequency == 0:
                self.update_gmm()

        return self.t, xy.flatten()[self.expl_dims], measure

    def update_gmm(self):
        scaled_data = self.scaler.fit_transform(self.data)

        self.gmm = GMM(n_components=self.n_components, covariance_type='full')
        self.gmm.fit(numpy.array(scaled_data))
        self.gmm_choice = self.gmm_interest()

    def gmm_interest(self):
        cov_t_c = numpy.array([self.gmm.covariances_[k, 0, -1]
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
