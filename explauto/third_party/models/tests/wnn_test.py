import testenv

import random
import numpy as np

import toolbox

import models
from models.forward.wnn import WeightedNNForwardModel
from models.inverse.wnn import WeightedNNInverseModel

# Test where datapoint are provided by group of k time the same point.

def test_fwdwnn_onepoint():
    """Test FwdWNN with only one point provided."""
    result = True

    max_dim = 5
    for i in range(5):
        n = random.randint(1, max_dim)
        m = random.randint(1, max_dim)
        k = random.randint(1, max_dim)
        sigma = random.uniform(0.1, 10)
        model = WeightedNNForwardModel(n, m, sigma = sigma, k = k)

        x = np.random.rand(n)
        y = np.random.rand(m)
        model.add_xy(x, y)

        for i in range(10):
            x = np.random.rand(n)
            ye = y
            yp = model.predict_y(x)
            check = np.allclose(ye, yp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', x, ye, yp)
            result = result and check

    return result

def test_fwdwnn_samepoint():
    """Test FwdWNN with one point provided multiple times."""
    result = True

    for i in range(25):
        n = random.randint(1, 3)
        m = random.randint(1, 3)
        k = random.randint(1, 20)
        sigma = random.uniform(0.1, 10)
        model = WeightedNNForwardModel(n, m, sigma = sigma, k = k)

        x = np.random.rand(n)
        y = np.random.rand(m)
        for i in range(random.randint(1, 20)):
            model.add_xy(x, y)

        for i in range(10):
            x = np.random.rand(n)
            ye = y
            yp = model.predict_y(x)
            check = np.allclose(ye, yp, rtol = 1e-5, atol = 1e-5)
            if not check:
                print('Error:', x, ye, yp)
            result = result and check

    return result

def test_fwdwnn_kpoints_group():
    """Test FwdWNN with group of k same points provided : all request should return
     point of the dataset."""
    result = True

    size = 30

    for i in range(20):
        n = random.randint(1, size)
        m = random.randint(1, size)
        k = random.randint(1, size)
        sigma = random.uniform(0.1, 10)
        model = WeightedNNForwardModel(n, m, sigma = sigma, k = k)

        ygroup = set()
        for i in range(random.randint(1, size)):
            x = np.random.rand(n)
            y = np.random.rand(m)
            ygroup.add(tuple(y))
            for i in range(k):
                model.add_xy(x, y)

        for i in range(10):
            x = np.random.rand(n)
            yp = model.predict_y(x)

            check = min([toolbox.dist(yp, y) for y in ygroup]) < 1e-10
            if not check:
                print('Error:', n, m, k, min([toolbox.dist(yp, y) for y in ygroup]), yp)
            result = result and check

    return result


def test_invwnn_onepoint():
    """Test InvWNN with only one point provided."""
    result = True

    for i in range(25):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        sigma = random.uniform(0.1, 10)
        model = WeightedNNInverseModel(n, m, sigma, k)

        x = np.random.rand(n)
        y = np.random.rand(m)
        model.add_xy(x, y)

        for i in range(10):
            y   = np.random.rand(m)
            xe = x
            xp = model.infer_x(y)[0]
            check = np.allclose(xe, xp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', n, m, y, xe, xp)
            result = result and check

    return result

def test_invwnn_samepoint():
    """Test InvWNN with one point provided multiple times."""

    result = True

    for i in range(25):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        sigma = random.uniform(0.1, 10)
        model = WeightedNNInverseModel(n, m, sigma, k)

        x = np.random.rand(n)
        y = np.random.rand(m)

        for i in range(random.randint(1, 20)):
            model.add_xy(x, y)

        for i in range(10):
            y   = np.random.rand(m)
            xe = x
            xp = model.infer_x(y)[0]
            check = np.allclose(xe, xp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', n, m, y, xe, xp)
            result = result and check

    return result

def test_invwnn_kpoints_group():
    """Test InvWNN with group of k same points provided : all request should return
     point of the dataset."""
    result = True

    for i in range(25):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        sigma = random.uniform(0.1, 10)
        model = WeightedNNInverseModel(n, m, sigma, k)

        xgroup = set()
        for i in range(random.randint(1, 20)):
            x = np.random.rand(n)
            y = np.random.rand(m)
            xgroup.add(tuple(x))
            for i in range(k):
                model.add_xy(x, y)

        for i in range(10):
            y   = np.random.rand(m)
            xp = model.infer_x(y)[0]
            check = min([toolbox.dist(xp, x) for x in xgroup]) < 1e-10
            if not check:
                print('Error:', n, m, y, xe, xp)
            result = result and check

    return result

tests = [test_fwdwnn_onepoint,
         test_fwdwnn_samepoint,
         test_fwdwnn_kpoints_group,
         test_invwnn_onepoint,
         test_invwnn_samepoint,
         test_invwnn_kpoints_group,
        ]

if __name__ == "__main__":
    print(("\033[1m%s\033[0m" % (__file__,)))
    for t in tests:
        print(('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__)))
