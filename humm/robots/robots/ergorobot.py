from math import pi, sin, cos
import pandas

# Constant values
a = 35
b = 82
c = 80
d = 65
e = 67
f = 63

def forwardkin_ergorobot(angles):
    t0, t1, t2, t3, t4, t5 = angles
    x =(0
        - b*sin(t1)*cos(t0)
        + 0
        + -c*sin(t1 + t2)*cos(t0) - d*sin(t0)*sin(t3) + d*cos(t0)*cos(t3)*cos(t1 + t2)
        + e*(-sin(t0)*sin(t3)*cos(t4) - sin(t4)*sin(t1 + t2)*cos(t0) + cos(t0)*cos(t3)*cos(t4)*cos(t1 + t2))
        + f*(-sin(t0)*sin(t3)*cos(t4 + t5) - sin(t1 + t2)*sin(t4 + t5)*cos(t0) + cos(t0)*cos(t3)*cos(t1 + t2)*cos(t4 + t5))
        )

    y =(0
        - b*sin(t0)*sin(t1)
        + 0
        + -c*sin(t0)*sin(t1 + t2) + d*sin(t0)*cos(t3)*cos(t1 + t2) + d*sin(t3)*cos(t0)
        + e*(-sin(t0)*sin(t4)*sin(t1 + t2) + sin(t0)*cos(t3)*cos(t4)*cos(t1 + t2) + sin(t3)*cos(t0)*cos(t4))
        + f*(-sin(t0)*sin(t1 + t2)*sin(t4 + t5) + sin(t0)*cos(t3)*cos(t1 + t2)*cos(t4 + t5) + sin(t3)*cos(t0)*cos(t4 + t5))
        )

    z =(a
        + b*cos(t1)
        + 0
        + c*cos(t1 + t2) + d*sin(t1 + t2)*cos(t3)
        + e*(sin(t4)*cos(t1 + t2) + sin(t1 + t2)*cos(t3)*cos(t4))
        + f*(sin(t1 + t2)*cos(t3)*cos(t4 + t5) + sin(t4 + t5)*cos(t1 + t2))
        )

    return (x, y, z)

class Ergorobot(object):
    """Forward model for an ergorobot stem
    #TODO : allow range to be expressed in integer between 0 and 1024
    """

    def __init__(self, bounds = None):
        """Initialize the model with function f
        @param bounds  ergorobot joint bounds.
        """
        self.m_feats  = tuple(range(-6, 0))
        self.m_bounds = bounds or 6*((-pi, pi),)
        assert len(self.m_bounds) == 6
        self.s_feats  = tuple(range(0, 3))
        self.name = 'Ergorobot'

    def __repr__(self):
        return self.name

    def execute_order(self, orderInM):
        """Return the effect"""
        return pandas.Series(forwardkin_ergorobot(list(orderInM[list(self.m_feats)])), index = self.s_feats)
