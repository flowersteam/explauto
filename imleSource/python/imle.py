import numpy

import _imle

d, D = 7, 3


class Imle(object):
    def __init__(self, **kwargs):
        f = lambda key, default: kwargs[key] if key in kwargs else default

        args = []

        args.append(f('alpha', 0.995))
        args.append(list(f('Psi0', [1] * D)))
        args.append(f('sigma0', 0.1))
        args.append(f('wsigma', 2.0**d))
        args.append(f('wSigma', 2.0**d))
        args.append(f('wNu', 0.0))
        args.append(f('wLambda', 0.1))
        args.append(f('wPsi', 0.0))
        args.append(f('p0', 0.1))
        args.append(f('multiValuedSignificance', 0.95))
        args.append(f('nSolMax', 8))

        param = _imle.ImleParam()
        param.set_param(*args)

        self._delegate = _imle.Imle(param)

    def update(self, z, x):
        if len(x) != D or len(z) != d:
            raise ValueError('check the inputs dimension')

        self._delegate.update(list(z), list(x))

    def predict(self, z):
        if len(z) != d:
            raise ValueError('check the inputs dimension')

        return numpy.array(self._delegate.predict(list(z)))

    def predict_inverse(self, x):
        if len(x) != D:
            raise ValueError('check the inputs dimension')

        return numpy.array(self._delegate.predict_inverse(list(x)))


    def get_joint_mu(self, k):
        """ The mean of the kth component of the joint GMM."""
        return numpy.array(self._delegate.get_joint_mu(k))

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
        l1 = list(numpy.random.randn(7))
        l2 = list(numpy.random.randn(3))

        i.update(l1, l2)

    print i.predict(range(7))
    print i.predict(range(7))
    print i.predict(range(7))

    print i.predict_inverse([4, 5, 6])

