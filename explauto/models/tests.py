from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import array, eye, pi, cos, linspace

from explauto.models.gmminf import GMM

def test_gmminf(value_y = 0., n_components=3):


    # Generate a dataset (noised cosinus)
    xlim = [-2, 2]
    x = linspace(xlim[0], xlim[1], 100)
    y = cos(x) + 0.1 * randn(*x.shape)
    data = array([x, y]).T

    # Fit a GMM P(XY). Our GMM class is a subclasse of sklearn.mixture.GMM, so you can use all the methods from this latter
    gmm = GMM(n_components=n_components, covariance_type='full')
    gmm.fit(data)

    # Infer the gmm P(X | Y=value_y) (the infered gmm is therefore 1D)
    gmm_inf = gmm.inference([1], [0], array(value_y))

    # PLot the result
    plt.figure()

    # Plot dataset, fitted gmm P(XY) and value_y
    ax = plt.subplot(211)
    ax.plot(x, y, '.')
    gmm.plot(ax)
    ax.plot(xlim, [value_y, value_y])
    ax.set_xlim(xlim)
    ax.set_xlabel()

    # Plot the infered GMM P(X | Y=value_y)
    ax = plt.subplot(212)
    gmm_inf.plot(ax)
    ax.set_xlim(xlim)

    plt.show()
