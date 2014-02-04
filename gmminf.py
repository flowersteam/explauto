import numpy
import sklearn.mixture

from numpy import ix_, array, all

from gaussian import Gaussian

class GMM(sklearn.mixture.GMM):
    def __init__(self, **kwargs):
        sklearn.mixture.GMM.__init__(self, **kwargs)
        self.in_dims = array([])
        self.out_dims = array([])

    def __iter__(self):
        for weight, mean, covar in zip(self.weights_, self.means_, self.covars_):
            yield (weight, mean, covar)

    def inference(self, in_dims, out_dims, value):
        new = not(all(array(in_dims)==self.in_dims) and all(array(out_dims)==self.out_dims))
        if new:
            print '1', array(in_dims), self.in_dims
            self.in_dims = numpy.array(in_dims)
            print '2', array(in_dims), self.in_dims
            self.out_dims = numpy.array(out_dims)		
            self.means_inf = numpy.zeros((self.n_components, len(self.out_dims)))
            self.covars_inf = numpy.zeros((self.n_components, len(self.out_dims), len(self.out_dims)))
            self.weights_inf = numpy.zeros((self.n_components,))
			
        if self.in_dims.size:
            for k, (weight_k, mean_k, covar_k) in enumerate(self):
                if new:
                    self.sig_in = covar_k[ix_(self.in_dims, self.in_dims)]
                    self.inin_inv = numpy.matrix(self.sig_in).I
                    self.out_in=covar_k[ix_(self.out_dims, self.in_dims)]
                    self.mu_in=mean_k[self.in_dims].reshape(-1,1)
                self.means_inf[k,:] = (mean_k[self.out_dims] + 
                            (self.out_in * 
                            self.inin_inv * 
                            (value - self.mu_in)).T)
                        
                self.covars_inf[k,:,:] = (covar_k[ix_(self.out_dims, self.out_dims)] - 
                self.out_in * 
                self.inin_inv * 
                covar_k[ix_(self.in_dims, self.out_dims)])
                self.weights_inf[k] = weight_k * Gaussian(self.mu_in.reshape(-1,), self.sig_in).normal(value.reshape(-1,))
            self.weights_inf /= sum(self.weights_inf)
        else:
            self.means_inf=self.means_[:,out_dims]
            self.covars_inf=self.covars_[ix_(range(self.n_components),self.out_dims,self.out_dims)]
            self.weights_inf=self.weights_
        res = GMM(n_components=self.n_components, 
                  covariance_type=self.covariance_type)
        res.weights_ = self.weights_inf
        res.means_ = self.means_inf
        res.covars_ = self.covars_inf
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
