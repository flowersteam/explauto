import numpy


class Gaussian(object):
    """
    Represents a single Gaussian probability
    density function.

    WARNING: some methods of this class may accept either one vector, as a (d,)
    shaped array, or many, as a (n, d) shaped array. For d = 1, do NOT use (n,)
    shaped array instead of (n, 1). The last formulation brings an ambiguity
    that is NOT handled.
    """

    def __init__(self, mu, sigma, inv_sigma=False):
        """
        Creates the Gaussian with the given parameters.
        @param mu : mean, given as (d,) matrix
        @param sigma : covariance matrix
        @param inv_sigma : boolean indicating if sigma is the inverse covariance matrix or not (default: False)
        """
        self.mu = mu
        if not inv_sigma:
            self.sigma = sigma
            self.inv = numpy.linalg.inv(self.sigma)
        else:
            self.sigma = numpy.linalg.inv(sigma)
            self.inv = self.sigma
        self.det = numpy.absolute(numpy.linalg.det(self.sigma))

    def generate(self, number=None):
        """Generates vectors from the Gaussian.

        @param number: optional, if given, generates more than one vector.

        @returns: generated vector(s), either as a one dimensional array
            (shape (d,)) if number is not set, or as a two dimensional array
            (shape (n, d)) if n is given as number parameter.
        """
        if number is None:
            return numpy.random.multivariate_normal(self.mu, self.sigma)
        else:
            return numpy.random.multivariate_normal(self.mu, self.sigma, number)

    def normal(self, x):
        """Returns the density of probability of x or the one dimensional
        array of all probabilities if many vectors are given.

        @param x : may be of (n,) shape.
        """
        return numpy.exp(self.log_normal(x))

    def log_normal(self, x):
        """
        Returns the log density of probability of x or the one dimensional
        array of all log probabilities if many vectors are given.

        @param x : may be of (n,) shape
        """
        d = self.mu.shape[0]
        xc = x - self.mu
        if len(x.shape) == 1:
            exp_term = numpy.sum(numpy.multiply(xc, numpy.dot(self.inv, xc)))
        else:
            exp_term = numpy.sum(numpy.multiply(xc, numpy.dot(xc, self.inv)), axis=1)

        return -.5 * (d * numpy.log(2 * numpy.pi) + numpy.log(self.det) + exp_term)

    def cond_gaussian(self, dims, v):
        """
        Returns mean  and variance of the conditional probability
        defined by a set of dimension and at a given vector.

        @param dims : set of dimension to which respect conditional
            probability is taken
        @param v : vector defining the position where the conditional
            probability is taken. v shape is defined by the size
            of the set of dims.
        """
        (d, c, b, a) = numpy.split_matrix(self.sigma, dims)
        (mu2, mu1) = numpy.split_vector(self.mu, dims)
        d_inv = numpy.linalg.inv(d)
        mu = mu1 + numpy.dot(numpy.dot(b, d_inv), v - mu2)
        sigma = a - numpy.dot(b, numpy.dot(d_inv, c))
        return Gaussian(mu, sigma)
    # TODO :  use a representation that allows different values of v
    #    without computing schur each time.

    def get_entropy(self):
        """Computes (analyticaly) the entropy of the Gaussian distribution.
        """
        dim = self.mu.shape[0]
        entropy = 0.5 * (dim * (numpy.log(2. * numpy.pi) + 1.) + numpy.log(self.det))
        return entropy

    def get_display_ellipse2D(self):
        from matplotlib.patches import Ellipse
        if self.mu.shape != (2,):
            raise ValueError('Not a 2 dimensional gaussian')

        (val, vect) = numpy.linalg.eig(self.sigma)
        el = Ellipse(self.mu,
                     3.5 * numpy.sqrt(val[0]),
                     3.5 * numpy.sqrt(val[1]),
                     180. * numpy.arctan2(vect[1, 0], vect[0, 0]) / numpy.pi,
                     fill=False,
                     linewidth=2)
        return el
