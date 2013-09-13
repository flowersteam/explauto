import numpy
import sklearn.mixture

from numpy import ix_

from gaussian import Gaussian

class GMM(sklearn.mixture.GMM):
    def __iter__(self):
        for weight, mean, covar in zip(self.weights_, self.means_, self.covars_):
            yield (weight, mean, covar)

    def inference(self, in_dims, out_dims, value):
        in_dims = numpy.array(in_dims)
        out_dims = numpy.array(out_dims)
    
        means = numpy.zeros((self.n_components, len(out_dims)))
        covars = numpy.zeros((self.n_components, len(out_dims), len(out_dims)))
        weights = numpy.zeros((self.n_components,))
        
        for k, (weight_k, mean_k, covar_k) in enumerate(self):
            sig_in = covar_k[ix_(in_dims, in_dims)]
            inin_inv = numpy.matrix(sig_in).I
            
            means[k,:] = (mean_k[out_dims] + 
                          (covar_k[ix_(out_dims, in_dims)] * 
                           inin_inv * 
                           (value - mean_k[ix_(in_dims)]).reshape(-1, 1)).T)
                    
            covars[k,:,:] = (covar_k[ix_(out_dims, out_dims)] - 
                             covar_k[ix_(out_dims, in_dims)] * 
                             inin_inv * 
                             covar_k[ix_(in_dims, out_dims)])
            
            weights[k] = weight_k * Gaussian(mean_k[in_dims], sig_in).normal(value)
            
        weights /= sum(weights)
        
        res = GMM(n_components=self.n_components, 
                  covariance_type=self.covariance_type)
        res.weights_ = weights
        res.means_ = means
        res.covars_ = covars
        return res

if __name__ == '__main__':
    gmm = GMM(n_components=100, covariance_type='full')

    obs = numpy.concatenate((numpy.random.randn(100, 10),
                            10 + numpy.random.randn(300, 10)))

    gmm.fit(obs)

    in_dims = numpy.arange(3)
    out_dims = numpy.arange(3, 10)
    value = 5 + numpy.random.randn(len(in_dims))
            
    newg = gmm.inference(in_dims, out_dims, value)