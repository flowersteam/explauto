import testenv

import random
import numpy as np

import toolbox

import models
from models.forward.avgnn import AverageNNForwardModel
from models.inverse.avgnn import AverageNNInverseModel

# Test where datapoint are provided by group of k time the same point.

def test_fwdavgnn_onepoint():
    """Test FwdAvgNN with only one point provided."""
    result = True

    for i in range(50):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNForwardModel(n, m, k = k)

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

def test_fwdavgnn_samepoint():
    """Test FwdAvgNN with one point provided multiple times."""
    result = True

    for i in range(50):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNForwardModel(n, m, k = k)

        x = np.random.rand(n)
        y = np.random.rand(m)
        for i in range(random.randint(1, 20)):
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

def test_fwdavgnn_kpoints():
    """Test FwdAvgNN with k point provided : all request should get the same prediction."""
    result = True

    for i in range(50):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNForwardModel(n, m, k = k)

        for i in range(k):
            x = np.random.rand(n)
            y = np.random.rand(m)
            model.add_xy(x, y)

        x   = np.random.rand(n)
        ypg = model.predict_y(x)


        for i in range(10):
            x = np.random.rand(n)
            ye = ypg
            yp = model.predict_y(x)
            check = np.allclose(ye, yp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', ye, yp)
            result = result and check

    return result

def test_fwdavgnn_kpoints_group():
    """Test FwdAvgNN with group of k same points provided : all request should return
     point of the dataset."""
    result = True

    for i in range(25):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNForwardModel(n, m, k = k)

        ygroup = set()
        for i in range(random.randint(1, 20)):
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
                print('Error:', x, yp)
            result = result and check

    return result


def test_invavgnn_onepoint():
    """Test InvAvgNN with only one point provided."""
    result = True

    for i in range(20):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNInverseModel(n, m, k)

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

def test_invavgnn_samepoint():
    """Test InvAvgNN with one point provided multiple times."""

    result = True

    for i in range(20):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNInverseModel(n, m, k)

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

def test_invavgnn_kpoints():
    """Test InvAvgNN with k point provided : all request should get the same prediction."""

    result = True

    for i in range(20):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNInverseModel(n, m, k)

        for i in range(k):
            x = np.random.rand(n)
            y = np.random.rand(m)
            model.add_xy(x, y)

        y   = np.random.rand(m)
        xpg = model.infer_x(y)

        for i in range(10):
            y   = np.random.rand(m)
            xe = xpg
            xp = model.infer_x(y)[0]
            check = np.allclose(xe, xp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', n, m, y, xe, xp)
            result = result and check

    return result


def test_invavgnn_kpoints_group():
    """Test InvAvgNN with group of k same points provided : all request should return
     point of the dataset."""
    result = True

    for i in range(20):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        k = random.randint(1, 20)
        model = AverageNNInverseModel(n, m, k)

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

tests = [test_fwdavgnn_onepoint,
         test_fwdavgnn_samepoint,
         test_fwdavgnn_kpoints,
         test_fwdavgnn_kpoints_group,
         test_invavgnn_onepoint,
         test_invavgnn_samepoint,
         test_invavgnn_kpoints,
         test_invavgnn_kpoints_group,
        ]

if __name__ == "__main__":
    print(("\033[1m%s\033[0m" % (__file__,)))
    for t in tests:
        print(('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__)))
