

import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gmminf import GMM
from gaussian import Gaussian


if __name__ == "__main__":
    mu1 = numpy.array([-3., 0.])
    mu2 = numpy.array([.33, numpy.pi])
    mu3 = numpy.array([5., -2.])
    sigma1 = numpy.eye(2)
    sigma2 = numpy.array([[3, 0], [0, .1]])
    sigma3 = numpy.array([[3, -2], [-1, 2]])

    weights = numpy.array([.5, .2, .3])


    gmm = GMM(n_components=3, covariance_type='full')
    gmm.means_ = numpy.array((mu1, mu2, mu3))
    gmm.covars_ = numpy.array((sigma1, sigma2, sigma3))
    gmm.weights_ = weights

    data = gmm.sample(1000)

    gmm2 = GMM(n_components=3, covariance_type='full')
    gmm2.fit(data)

    # Display
    COLORS = ['r', 'g', 'b']
    orig_colors = COLORS
    found_colors = COLORS

    plt.figure()

    s1 = plt.subplot(311)
    s1.scatter(data[:, 0], data[:, 1], color='b')

    els = gmm.get_display_ellipses2D(COLORS)
    for el in els:
        s1.add_patch(el)

    s2 = plt.subplot(312)
    s2.scatter(data[:, 0], data[:, 1], color='b')

    els2 = gmm2.get_display_ellipses2D(COLORS)
    for el in els2:
        s2.add_patch(el)

    # Test GMR
    y_cond0 = (-4.,)
    y_cond1 = (0,)
    y_cond2 = (2.5,)
    x_inf = -10.
    x_sup = 15.

    cond0 = gmm2.inference((1,), (0,), y_cond0)
    cond1 = gmm2.inference((1,), (0,), y_cond1)
    cond2 = gmm2.inference((1,), (0,), y_cond2)

    s3 = plt.subplot(313)
    for (y_cond, col) in zip([y_cond0, y_cond1, y_cond2], COLORS):
        line = Line2D([x_inf, x_sup],
                [y_cond, y_cond],
                color = col,
                linewidth = 2)
        s1.add_line(line)
    x = numpy.arange(x_inf, x_sup, .1)
    for (cond, col) in zip([cond0, cond1, cond2], COLORS) :
        plt.plot(x, numpy.exp(cond.score(x[:, numpy.newaxis])), color=col)


    plt.show()