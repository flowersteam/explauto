import numpy
import sklearn.mixture

from numpy import ix_, array

from gaussian import Gaussian

class GMM(sklearn.mixture.GMM):
    def __init__(self, **kwargs):
        sklearn.mixture.GMM.__init__(self, **kwargs)
        self.in_dims = array([])
        self.out_dims = array([])

    def __iter__(self):
        for weight, mean, covar in zip(self.weights_, self.means_, self.covars_):
            yield (weight, mean, covar)

    def probability(self, value):
        p =0.
        for k, (w, m, c) in enumerate(self):
            p += w * Gaussian(m.reshape(-1,), c).normal(value.reshape(-1,))
        return p

    def sub_gmm(self, inds_k):
        gmm = GMM(n_components=len(inds_k), covariance_type=self.covariance_type)
        gmm.weights, gmm.means_, gmm.covars_ = self.weights_[inds_k], self.means_[inds_k,:], self.covars_[inds_k,:,:]
        gmm.weights = gmm.weights / gmm.weights.sum()
        return gmm

    def inference(self, in_dims, out_dims, value=None):
        in_dims = numpy.array(in_dims)
        out_dims = numpy.array(out_dims)
        value = numpy.array(value)
    
        means = numpy.zeros((self.n_components, len(out_dims)))
        covars = numpy.zeros((self.n_components, len(out_dims), len(out_dims)))
        weights = numpy.zeros((self.n_components,))
        
        if in_dims.size:
            for k, (weight_k, mean_k, covar_k) in enumerate(self):
                sig_in = covar_k[ix_(in_dims, in_dims)]
                inin_inv = numpy.matrix(sig_in).I
                out_in=covar_k[ix_(out_dims, in_dims)]
                mu_in=mean_k[in_dims].reshape(-1,1)
                        
                means[k,:] = (mean_k[out_dims] + 
                            (out_in * 
                            inin_inv * 
                            (value - mu_in)).T)
                        
                covars[k,:,:] = (covar_k[ix_(out_dims, out_dims)] - 
                                out_in * 
                                inin_inv * 
                                covar_k[ix_(in_dims, out_dims)])
                weights[k] = weight_k * Gaussian(mu_in.reshape(-1,), sig_in).normal(value.reshape(-1,))
            weights /= sum(weights)
        else:
            means=self.means_[:,out_dims]
            covars=self.covars_[ix_(range(self.n_components),out_dims,out_dims)]
            weights=self.weights_
        res = GMM(n_components=self.n_components, 
                  covariance_type=self.covariance_type)
        res.weights_ = weights
        res.means_ = means
        res.covars_ = covars
        return res

    def get_display_ellipses2D(self, colors):
        from matplotlib.patches import Ellipse

        ellipses = []

        for i, (weight, mean, covar) in enumerate(self):
            (val, vect) = numpy.linalg.eig(covar)

            el = Ellipse(mean,
                         3.5 * numpy.sqrt(val[0]),
                         3.5 * numpy.sqrt(val[1]),
                         180. * numpy.arctan2(vect[1, 0], vect[0, 0]) / numpy.pi,
                         fill=False,
                         linewidth=2)

            el.set_facecolor(colors[i])
            el.set_fill(True)
            el.set_alpha(0.5)

            ellipses.append(el)

        return ellipses

if __name__ == '__main__':
    gmm = GMM(n_components=100, covariance_type='full')

    obs = numpy.concatenate((numpy.random.randn(100, 10),
                            10 + numpy.random.randn(300, 10)))

    gmm.fit(obs)

    in_dims = numpy.arange(3)
    out_dims = numpy.arange(3, 10)
    value = 5 + numpy.random.randn(len(in_dims))
            
    newg = gmm.inference(in_dims, out_dims, value)
