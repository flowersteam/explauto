import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
# imle_path = os.path.join(path, 'IMLEv1.90/python')
#imle_path = os.path.join(path, 'imleSource/python')
#sys.path.append(imle_path)

try:
    import imle
except ImportError:
    print 'To use this model, you have to install IMLE first'
    print 'Please check the doc.'
    raise

# print imle.__file__

from numpy import zeros, ones

from .gmminf import GMM


class Imle(imle.Imle):
    def to_gmm(self):
        n = self.number_of_experts
        gmm = GMM(n_components=n, covariance_type='full')
        gmm.means_ = zeros((n, self.d+self.D))
        gmm.covars_ = zeros((n, self.d+self.D, self.d+self.D))

        for k in range(n):
            gmm.means_[k, :] = self.get_joint_mu(k)
            gmm.covars_[k, :, :] = self.get_joint_sigma(k)
        gmm.weights_ = (1.*ones((n,)))/n
        return gmm
