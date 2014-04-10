import importlib
import numpy
import sys
import os

#import _imle
#from .gmminf import GMM

#d, D = 2, 1

imle_path = '..'

def load_imle(imle_path, d, D):
    bp = os.path.dirname(__file__)
    bp = bp if bp else '.'
    p = os.path.join(bp, imle_path, 'build', '{}_{}'.format(d, D), 'lib')

    if not os.path.exists(p):
        raise ImportError('Compile IMLE for dim ({}, {}) first, by running "python compile.py {} {}" in {}'.format(d, D, d, D, os.path.abspath(os.path.join(bp,imle_path))))

    sys.path.insert(0, p)
    _imle = importlib.import_module('_imle_' + str(d) + '_' + str(D))
    del sys.path[0]

    return _imle


class Imle(object):
    def __init__(self,  imle_path=imle_path, **kwargs):
        f = lambda key, default: kwargs[key] if key in kwargs else default

    	self.d = kwargs['in_ndims']
    	self.D = kwargs['out_ndims']
    	self._imle = load_imle(imle_path, self.d, self.D)

        args = []
        args.append(f('alpha', 0.995))
        args.append(list(f('Psi0', [1] * self.D)))
        args.append(f('sigma0', 0.1))
        args.append(f('wsigma', 2.0**self.d))
        args.append(f('wSigma', 2.0**self.d))
        args.append(f('wNu', 0.0))
        args.append(f('wLambda', 0.1))
        args.append(f('wPsi', 2.0**self.d))
        args.append(f('p0', 0.1))
        args.append(f('multiValuedSignificance', 0.8))
        args.append(f('nSolMax', 8))

	#exec 'import _'+str(in_ndims)+'_'+str(out_ndims)
        param = self._imle.ImleParam()
        param.set_param(*args)

        self._delegate = self._imle.Imle(param)

    def update(self, z, x):
        if len(x) != self.D or len(z) != self.d:
            raise ValueError('check the inputs dimension', len(x), self.D, len(z), self.d)

        self._delegate.update(list(z), list(x))

    def predict(self, z):
        if len(z) != self.d:
            raise ValueError('check the inputs dimension')

        return numpy.array(self._delegate.predict(list(z))), \
            numpy.array(self._delegate.getPredictionVar()), \
            numpy.array(self._delegate.getPredictionWeight()), \
            numpy.array(self._delegate.getPredictionJacobian()) 

    def predict_inverse(self, x):
        if len(x) != self.D:
            raise ValueError('check the inputs dimension', len(x), self.D)

        return numpy.array(self._delegate.predict_inverse(list(x))), \
            numpy.array(self._delegate.getPredictionVar()), \
            numpy.array(self._delegate.getPredictionWeight())

    def get_prediction_weight(self):
        return self._delegate.getPredictionWeight()

    def get_joint_mu(self, k):
        """ The mean of the kth component of the joint GMM."""
        if k >= self.number_of_experts:
            raise ValueError('check the expert indice')
        return numpy.array(self._delegate.get_joint_mu(k))

    def get_joint_sigma(self, k):
        """ The covariance matrix of the kth component of the joint GMM."""
        if k >= self.number_of_experts:
            raise ValueError('check the expert indice')
        Sigma=self.get_sigma(k)
        Lambda=self.get_lambda(k)
        SigmaLambdaT=Sigma.dot(Lambda.T)
        #LambdaSigma=Lambda.dot(Sigma)
        return numpy.vstack((numpy.hstack((Sigma, SigmaLambdaT)), numpy.hstack((SigmaLambdaT.T, numpy.diag(self.get_psi(k)) + Lambda.dot(SigmaLambdaT)))))

    #def to_gmm(self):
        #n=self.number_of_experts
        #gmm=GMM(n_components=n, covariance_type='full')
        #gmm.means_=numpy.zeros((n,self.d+self.D))
        #gmm.covars_=numpy.zeros((n,self.d+self.D,self.d+self.D))
        #for k in range(n):
            #gmm.means_[k,:]=self.get_joint_mu(k)
            #gmm.covars_[k,:,:]=self.get_joint_sigma(k)
        #gmm.weights_= (1.*numpy.ones((n,)))/n
        #return gmm


    def get_sigma(self, k):
        """ The input covariance matrix of the kth component."""
        if k >= self.number_of_experts:
            raise ValueError('check the expert indice')
        return numpy.linalg.inv(numpy.array(self._delegate.get_inv_sigma(k)))

    def get_lambda(self, k):
        """ The linear transformation of the kth component."""
        if k >= self.number_of_experts:
            raise ValueError('check the expert indice')
        return numpy.array(self._delegate.get_lambda(k))

    def get_psi(self, k):
        """ The output noise of the kth component."""
        if k >= self.number_of_experts:
            raise ValueError('check the expert indice')
        return numpy.array(self._delegate.get_psi(k))

    @property
    def number_of_experts(self):
        """ The number_of_experts property."""
        return self._delegate.get_number_of_experts()

    @property
    def psi0(self):
        """ The foo property."""
        return self._delegate.get_psi0()

    @property
    def wPsi(self):
        """ The wPsi property."""
        return self._delegate.get_wPsi()


if __name__ == '__main__':
    i = Imle()

    for _ in range(100):
        l1 = list(numpy.random.randn(i.d))
        l2 = list(numpy.random.randn(i.D))

        i.update(l1, l2)

    print i.predict(range(7))
    print i.predict(range(7))
    print i.predict(range(7))

    print i.predict_inverse([4, 5, 6])

