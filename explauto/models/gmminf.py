import sklearn.mixture
from numpy.linalg import inv, eig
from numpy import ix_, array, inf, sqrt, linspace, zeros, arctan2, matrix, pi

from .gaussian import Gaussian


def schur_complement(mat, row, col):
    """ compute the schur complement of the matrix block mat[row:,col:] of the matrix mat """
    a = mat[:row, :col]
    b = mat[:row, col:]
    c = mat[row:, :col]
    d = mat[row:, col:]
    return a - b.dot(d.I).dot(c)


def conditional(mean, covar, dims_in, dims_out, covariance_type='full'):
    """ Return a function f such that f(x) = p(dims_out | dims_in = x) (f actually returns the mean and covariance of the conditional distribution
    """
    in_in = covar[ix_(dims_in, dims_in)]
    in_out = covar[ix_(dims_in, dims_out)]
    out_in = covar[ix_(dims_out, dims_in)]
    out_out = covar[ix_(dims_out, dims_out)]
    in_in_inv = inv(in_in)
    out_in_dot_in_in_inv = out_in.dot(in_in_inv)

    cond_covar = out_out - out_in_dot_in_in_inv.dot(in_out)
    cond_mean = lambda x: mean[dims_out] + out_in_dot_in_in_inv.dot(x - mean[dims_in])
    return lambda x: [cond_mean(x), cond_covar]


class GMM(sklearn.mixture.GMM):
    def __init__(self, **kwargs):
        sklearn.mixture.GMM.__init__(self, **kwargs)
        self.in_dims = array([])
        self.out_dims = array([])

    def __iter__(self):
        for weight, mean, covar in zip(self.weights_, self.means_, self.covars_):
            yield (weight, mean, covar)

    def probability(self, value):
        p = 0.
        for k, (w, m, c) in enumerate(self):
            p += w * Gaussian(m.reshape(-1,), c).normal(value.reshape(-1,))
        return p

    def sub_gmm(self, inds_k):
        gmm = GMM(n_components=len(inds_k), covariance_type=self.covariance_type)

        gmm.weights_, gmm.means_, gmm.covars_ = (self.weights_[inds_k],
                                                self.means_[inds_k, :],
                                                self.covars_[inds_k, :, :])
        gmm.weights_ = gmm.weights_ / gmm.weights_.sum()
        return gmm

    def conditional(self, in_dims, out_dims):
        conditionals = []

        for k, (weight_k, mean_k, covar_k) in enumerate(self):
            conditionals.append(conditional(mean_k, covar_k,
                                            in_dims, out_dims,
                                            self.covariance_type))

        cond_weights = lambda v: [(weight_k * Gaussian(mean_k[in_dims].reshape(-1,),
                                  covar_k[ix_(in_dims, in_dims)]).normal(v.reshape(-1,)))
                                  for k, (weight_k, mean_k, covar_k) in enumerate(self)]

        def res(v):
            gmm = GMM(n_components=self.n_components,
                      covariance_type=self.covariance_type,
                      random_state=self.random_state, thresh=self.thresh,
                      min_covar=self.min_covar, n_iter=self.n_iter, n_init=self.n_init,
                      params=self.params, init_params=self.init_params)
            gmm.weights_ = cond_weights(v)
            means_covars = [f(v) for f in conditionals]
            gmm.means_ = array([mc[0] for mc in means_covars]).reshape(self.n_components,
                                                                       -1)
            gmm._set_covars(array([mc[1] for mc in means_covars]))
            return gmm

        return res

        self.in_dims = array(in_dims)
        self.out_dims = array(out_dims)
        means = zeros((self.n_components, len(out_dims)))
        covars = zeros((self.n_components, len(out_dims), len(out_dims)))
        weights = zeros((self.n_components,))
        sig_in = []
        inin_inv = []
        out_in = []
        mu_in = []
        for k, (weight_k, mean_k, covar_k) in enumerate(self):
            sig_in.append(covar_k[ix_(in_dims, in_dims)])
            inin_inv.append(matrix(sig_in).I)
            out_in.append(covar_k[ix_(out_dims, in_dims)])
            mu_in.append(mean_k[in_dims].reshape(-1, 1))

            means[k, :] = (mean_k[out_dims] +
                           (out_in *
                            inin_inv *
                            (value - mu_in)).T)

            covars[k, :, :] = (covar_k[ix_(out_dims, out_dims)] -
                               out_in *
                               inin_inv *
                               covar_k[ix_(in_dims, out_dims)])
            weights[k] = weight_k * Gaussian(mu_in.reshape(-1,),
                                             sig_in).normal(value.reshape(-1,))
        weights /= sum(weights)

        def p(value):
            # hard copy of useful matrices local to the function
            pass

        return p

    def inference(self, in_dims, out_dims, value=None):
        """ Perform Bayesian inference on the gmm. Let's call V = V1...Vd the d-dimensional space on which the current GMM is defined, such that it represents P(V). Let's call X and Y to disjoint subspaces of V, with corresponding dimension indices in ran. This method returns the GMM for P(Y | X=value).

        :param list in_dims: the dimension indices of X (a subset of range(d)). This can be the empty list if one want to compute the marginal P(Y).

        :param list out_dims: the dimension indices of Y (a subset of range(d), without intersection with in_dims).

        :param numpy.array value: the value of X for which one want to compute the conditional (ignored of in_dims=[]).

        :returns: the gmm corresponding to P(Y | X=value) (or to P(Y) if in_dims=[])

        .. note:: For example, if X = V1...Vm and Y = Vm+1...Vd, then P(Y | X=v1...vm) is returned by self.inference(in_dims=range(m), out_dims=range(m, d), array([v1, ..., vm])).
        """

        if self.covariance_type != 'diag' and self.covariance_type != 'full':
            raise ValueError("covariance type other than 'full' and 'diag' not allowed")
        in_dims = array(in_dims)
        out_dims = array(out_dims)
        value = array(value)

        means = zeros((self.n_components, len(out_dims)))
        covars = zeros((self.n_components, len(out_dims), len(out_dims)))
        weights = zeros((self.n_components,))
        if in_dims.size:
            for k, (weight_k, mean_k, covar_k) in enumerate(self):

                sig_in = covar_k[ix_(in_dims, in_dims)]
                inin_inv = matrix(sig_in).I
                out_in = covar_k[ix_(out_dims, in_dims)]
                mu_in = mean_k[in_dims].reshape(-1, 1)
                means[k, :] = (mean_k[out_dims] +
                               (out_in *
                                inin_inv *
                                (value.reshape(-1, 1) - mu_in)).T)
                if self.covariance_type == 'full':
                    covars[k, :, :] = (covar_k[ix_(out_dims, out_dims)] -
                                       out_in *
                                       inin_inv *
                                       covar_k[ix_(in_dims, out_dims)])
                elif self.covariance_type == 'diag':
                    covars[k, :] = covar_k[out_dims]

                weights[k] = weight_k * Gaussian(mu_in.reshape(-1,),
                                                 sig_in).normal(value.reshape(-1,))
            weights /= sum(weights)
        else:
            means = self.means_[:, out_dims]
            if self.covariance_type == 'full':
                covars = self.covars_[ix_(range(self.n_components), out_dims, out_dims)]
            if self.covariance_type == 'diag':
                covars = self.covars_[ix_(range(self.n_components), out_dims)]
            weights = self.weights_

        res = GMM(n_components=self.n_components,
                  covariance_type=self.covariance_type)
        res.weights_ = weights
        res.means_ = means
        res.covars_ = covars
        return res

    def ellipses2D(self, colors):
        from matplotlib.patches import Ellipse

        ellipses = []

        for i, ((weight, mean, _), covar) in enumerate(zip(self, self._get_covars())):
            (val, vect) = eig(covar)

            el = Ellipse(mean,
                         3.5 * sqrt(val[0]),
                         3.5 * sqrt(val[1]),
                         180. * arctan2(vect[1, 0], vect[0, 0]) / pi,
                         fill=False,
                         linewidth=2)

            el.set_facecolor(colors[i])
            el.set_fill(True)
            el.set_alpha(0.5)

            ellipses.append(el)

        return ellipses

    def ellipses3D(self):
        from ..utils.ellipsoid import ellipsoid_3d
        ellipsoids = []
        for k, (weight_k, mean_k, covar_k) in enumerate(self):
            ellipsoids.append(ellipsoid_3d(mean_k, covar_k))
        return ellipsoids

    def plot(self, ax, label=False):
        self.plot_projection(ax, range(self.means_.shape[1]), label)

    def plot_projection(self, ax, dims, label=False):
        COLORS = self.weights_ / max(self.weights_)
        COLORS = [str(c) for c in COLORS]
        # COLORS = ['r', 'g', 'b', 'k', 'm']*10000
        gmm_proj = self.inference([], dims, [])
        if len(dims) == 1:
            x_min, x_max = inf, -inf
            for w, m, c in self:
                x_min = min(x_min, m[0] - 3. * sqrt(c[0, 0]))
                x_max = max(x_max, m[0] + 3. * sqrt(c[0, 0]))
            x = linspace(x_min, x_max, 1000)
            p = [self.probability(xx) for xx in x]
            ax.plot(x, p)

        elif len(dims) == 2:
            els = gmm_proj.ellipses2D(COLORS)
            for el in els:
                ax.add_patch(el)
        elif len(dims) == 3:
            ellipsoids = gmm_proj.ellipses3D()
            for x, y, z in ellipsoids:
                ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
        else:
            print "Can only print 2D or 3D ellipses"
        if label:
            for k, (w, m, c) in enumerate(gmm_proj):
                ax.text(* tuple(m), s=str(k))
        ax.axis('tight')
