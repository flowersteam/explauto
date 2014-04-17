import testenv

import random
random.seed(0)
import numpy as np
#import np.random
np.random.seed(0)

import models
from models.forward.lwr import LWLRForwardModel

def random_linear(n, m):
    """Create a random linear function from R^n -> R^m"""
    M = np.random.rand(n,m)
    return lambda x : np.dot(np.array(x), M).ravel()

def test_lwr1D_linear():
    """Simplest test possible (well, not quite, but close)."""
    result = True

    f = lambda x : 2.0*x
    model = LWLRForwardModel(1, 1, k = 3, sigma = 0.1)

    for i in range(10):
        x = np.random.rand(1)
        y = f(x)
        #print x, y
        model.add_xy(x, y)

    for i in range(10):
        x = np.random.rand(1).ravel()
        y = f(x)
        yp = model.predict_y(x)
        #print 'y ', y
        #print 'yp', yp
        result = result and np.allclose(y, yp, rtol = 1e-5, atol = 1e-5)

    return result


def test_lwr_linear():
    """Test LWLR on random linear models of dimensions from 1 to 20.
     It should return exact results, give of take floating point imprecisions."""
    result = True

    for i in range(20):
        n = random.randint(1, 20)
        m = random.randint(1, 5)
        f = random_linear(n, m)
        model = LWLRForwardModel(n, m, 1.0)

        for i in range(2*n):
            x = np.random.rand(n)
            y = f(x)
            model.add_xy(x, y)

        for i in range(10):
            x = np.random.rand(n).ravel()
            y = f(x)
            yp = model.predict_y(x)
            #print 'y ', y
            #print 'yp', yp
            result = result and np.allclose(y, yp, rtol = 1e-10, atol = 1e-10)

    return result

#TODO : move to another file to avoid crash when robots import does not work.

import models.testbed
import robots
import robots.fun

def test_lwr_linear_testbed():
    """Same test as previously, this time using testbed api."""
    result = True

    for i in range(20):
        n = random.randint(1, 20)
        m = random.randint(1, 20)
        f = random_linear(n, m)
        robot = robots.fun.RobotFunction(f, n, m)
        model = LWLRForwardModel(n, m, 1.0)

        for i in range(2*n):
            x = np.random.rand(n)
            y = f(x)
            model.add_xy(x, y)

        tb = models.testbed.Testbed(robot, fmodel = model)
        tb.uniform_motor(10)
        errors = tb.run_forward()
        result = result and sum(errors) < len(errors)*1e-10

    return result

def test_lwr_quadratic():
    """Test LWLR on 2nd order polynomial models of dimensions from 1 to 1.
     (should pass most of the time)"""
    result = True

    for i in range(20):

        a = random.random()
        b = random.random()
        c = random.random()
        f = lambda x: [a*x[0]*x[0] + b*x[0] + c]
        model = LWLRForwardModel(1, 1, 0.1)
        robot = robots.fun.RobotFunction(f, 1, 1)

        for i in range(100):
            x = np.random.rand(1)
            y = f(x)
            model.add_xy(x, y)

        tb = models.testbed.Testbed(robot, fmodel = model)
        tb.uniform_motor(20)
        errors = tb.run_forward()
        result = result and sum(errors) < len(errors)*1e-3

        # for i in xrange(10):
        #     x = np.random.rand(1)
        #     y = f(x)
        #     yp = model.predict_y(x)
        #     # import toolbox
        #     # print toolbox.dist(y, yp)
        #     result = result and np.allclose(y, yp, rtol = 1e-1, atol = 1e-1)

    return result



tests = [test_lwr1D_linear,
         test_lwr_linear,
         test_lwr_linear_testbed,
         test_lwr_quadratic
        ]

if __name__ == "__main__":
    print(("\033[1m%s\033[0m" % (__file__,)))
    for t in tests:
        print(('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__)))