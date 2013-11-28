# The action/perception cycle is organized as follow :
# - a motor order is send as an pandas.Series, indexed by features.
# - sensory datapoint (pandas.Series) out of sensory primitives are returned

try:
    import pandas
    pandas_available = True
except ImportError:
    pandas_available = False

from . import robot

class Prism(robot.Robot):

    def __init__(self):
        # angle, torque
        self.m_feats       = (-1, -2)
        self.m_bounds      = ((0.0, 180.0), (0.0, 5.0)) # Motor bounds
        self.s_feats       = (0,) # distance
        self.Sbounds      = ((0, 20.0),)
        self.Sresolutions = (25,)
        self.name         = 'Prism'

    def set_env(self, pos, mass, friction):
        assert len(pos) == 2
        self.pos  = pos
        self.mass = mass
        self.friction = friction

    def execute_order(self, order):
        """Return the effect"""
        x = self._pre_x(order)
        angle, torque = x
        if self.pos[0] <= angle <= self.pos[1]:
            y = torque**2*(1-self.friction)/self.mass # to be improved
        else:
            y = 0.0
        return self._post_y([y], x)

class Prism1(Prism):
    def __init__(self):
        Prism.__init__(self)
        self.set_env((60, 70), 1.0, 0.25)

class Prism2(Prism):
    def __init__(self):
        Prism.__init__(self)
        self.set_env((60, 70), 4.0, 0.61)

class Prism3(Prism):
    def __init__(self):
        Prism.__init__(self)
        self.set_env((110, 120), 1.0, 0.25)

class Prism4(Prism):
    def __init__(self):
        Prism.__init__(self)
        self.set_env((60, 70), 1.0, 0.25)
