from gaussian import Gaussian
from numpy import linspace, array, arange, tile, dot

class BasisFunctions(object):
    def __init__(self, n_basis, duration, dt, sigma):
        self.n_basis = n_basis
        means = linspace(0, duration, n_basis) 
        variances = duration / (sigma * n_basis)**2
        gaussians = [Gaussian(array([means[k]]), array([[variances]])) for k in range(len(means))] 
        self.x = arange(0., duration, dt)
        y = array([gaussians[k].normal(self.x.reshape(-1,1)) for k in range(len(means))] ) 
        self.z = y / tile(sum(y,0), (n_basis,1))
    def trajectory(self, weights):
        return dot(weights, self.z)
