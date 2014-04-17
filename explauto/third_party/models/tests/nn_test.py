import testenv

import random
import numpy as np

import models
from models.forward.nn import NNForwardModel
from models.inverse.nn import NNInverseModel


def test_fwdnn_lattice():
    """Test NN on a lattice dataset, where expected result can be computed directly."""
    result = True

    for i in range(10):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        f = lambda x: (n*x[0], m*x[0])
        model = NNForwardModel(1, 2)

        for i in range(11):
            x = [i]
            y = f(x)
            model.add_xy(x, y)

        for i in range(10):
            x = [random.uniform(0, 10)]
            ye = [n*round(x[0]), m*round(x[0])]
            yp = model.predict_y(x)
            check = np.allclose(ye, yp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', x, ye, yp)
            result = result and check

    return result

def test_invnn_lattice():
    """Test Inverse NN on a lattice dataset, where expected result can be computed directly."""
    result = True

    for i in range(10):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        f = lambda x: (n*x[0], m*x[1])
        model = NNInverseModel(2, 2)

        for i in range(11):
            for j in range(11):
                x = [i, j]
                y = f(x)
                model.add_xy(x, y)

        for i in range(10):
            y = [random.uniform(0, 10*n), random.uniform(0, 10*m)]
            xe = [round(y[0]/n), round(y[1]/m)]
            xp = model.infer_x(y)[0]
            check = np.allclose(xe, xp, rtol = 1e-10, atol = 1e-10)
            if not check:
                print('Error:', n, m, y, xe, xp)
            result = result and check

    return result


tests = [test_fwdnn_lattice,
         test_invnn_lattice]

if __name__ == "__main__":
    print(("\033[1m%s\033[0m" % (__file__,)))
    for t in tests:
        print(('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__)))
