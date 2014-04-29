import numpy as np

from scipy import stats


def rand_bounds(bounds, n=1):
    widths = np.tile(bounds[1, :] - bounds[0, :], (n, 1))
    return widths * np.random.rand(n, bounds.shape[1]) + np.tile(bounds[0, :], (n, 1))


def bounds_min_max(v, mins, maxs):
    res = np.minimum(v, maxs)
    res = np.maximum(res, mins)
    return res


def discrete_random_draw(data, nb=1):
    ''' Code from Steve Nguyen'''
    data = np.array(data)
    if not data.any():
        data = np.ones_like(data)
    data = data/data.sum()
    xk = np.arange(len(data))
    custm = stats.rv_discrete(name='custm', values=(xk, data))
    return custm.rvs(size=nb)


def rk4(x, h, y, f):
    """
        Code from http://doswa.com/2009/04/21/improved-rk4-implementation.html
        :param int x: curent time
        :param float h: time step (dt)
        :param list y: [position, velocity]
        :params function f: velocity, accelation[x+h] = f(x, state)
        :return int t, list state: x+h, next state
    """
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5*h, y + 0.5*k1)
    k3 = h * f(x + 0.5*h, y + 0.5*k2)
    k4 = h * f(x + h, y + k3)
    return x + h, y + (k1 + 2*(k2 + k3) + k4)/6.0
