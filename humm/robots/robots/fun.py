# Functions for creating synthetic sensorimotor spaces of controlled complexity

import math
import pandas


class RobotFunction(object):
    """Create a robot from a function"""

    def __init__(self, f, dimM, dimS, bounds = None):
        """Initialize the model with function f
        @param f       function taking at most one positional argument,
                       an array of float of dimension dimM, and returning
                       an array of float of dimension dimS
        @param bounds  describe the support of the function, as an hyperrectangle.
                       if not provided, each float will be constrained between
                       0 and 1.
        """
        self.f       = f
        self.m_feats  = tuple(range(-dimM, 0))
        self.m_bounds = bounds or dimM*((0.0, 1.0),)
        assert len(self.m_bounds) == dimM
        self.s_feats  = tuple(range(0, dimS))

    def execute_order(self, orderInM):
        """Return the effect"""
        return pandas.Series(self.f(list(orderInM[list(self.m_feats)])), index = self.s_feats)


class Sinus1D(object):
    """A sum of same period sinus waves."""

    def __init__(self, dimM = 2, complexity = 1):
        """
        @param complexity  the higher, the higher the complexity
        """
        self.m_feats  = tuple(range(-dimM, 0))
        self.m_bounds = dimM*((0.0, 1.0),)
        self.s_feats  = (0,)
        self.complexity = complexity
        self.conf    = {'dimM': dimM, 'complexity': complexity}

    def check_bounds(self, mfv):
        assert 1.00000001 > mfv > -0.000001, "Error: motor data out of range 0 !< {:.8f} !< 1".format(mfv)
        return min(1.0, max(0.0, mfv))

    def execute_order(self, orderInM):
        """Return the effect"""
        e = 0
        for mf in self.m_feats:
            mfv = orderInM.get(mf)
            mfv = self.check_bounds(mfv)
            e += math.sin(2*math.pi*self.complexity*mfv)
        return pandas.Series([e], index = self.s_feats)

    def __repr__(self):
        return "Sinus1D(dimM=%i, complexity=%s)" % (self.conf['dimM'], self.conf['complexity'])