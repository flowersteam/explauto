""" Learner is a wrapper class that regroups a :py:class:`Dataset`, a :py:class:`ForwardModel`, and an :py:class:`InverseModel`.
    It is the easiest way to use models, and is flexible enough to be initialized from robots, Forward and InverseModel instances.

    The class accept any from of iterable as input and ouput vector, as well as pandas.Series.
    In the latter case, the index should match either self.Mfeats or self.Sfeats.
"""

import numpy as np
import pandas

from . import forward
fwdclass = {'NN'     : forward.NNForwardModel,
            'WNN'    : forward.WeightedNNForwardModel,
            'ES-WNN' : forward.ESWNNForwardModel,
            'AvgNN'  : forward.AverageNNForwardModel,
            'LWLR'   : forward.LWLRForwardModel,
            'ES-LWLR': forward.ESLWLRForwardModel,
            'Rnd'    : forward.RandomForwardModel
           }

from . import inverse
invclass = {'NN'       : inverse.NNInverseModel,
            'WNN'      : inverse.WeightedNNInverseModel,
            'ES-WNN'   : inverse.ESWNNInverseModel,
            'AvgNN'    : inverse.AverageNNInverseModel,
            'BFGS'     : inverse.BFGSInverseModel,
            'L-BFGS-B' : inverse.BFGSInverseModel,
            'COBYLA'   : inverse.COBYLAInverseModel,
            'CMAES'    : inverse.CMAESInverseModel,
            'Rnd'      : inverse.RandomInverseModel
           }

class Learner(object):

    # Creation routines

    @classmethod
    def from_robot(cls, robot, fwd = 'LWLR', inv = 'L-BFGS-B', **kwargs):
        """ Create a learner from a robot instance. """
        return cls(robot.m_feats, robot.s_feats, robot.m_bounds, fwd = fwd, inv = inv, **kwargs)

    @classmethod
    def from_robot_and_fmodel(cls, robot, fmodel, inv = 'L-BFGS-B', **kwargs):
        """ Create a learner from a robot instance and a forward model

            :arg fmodel:  a FowardModel instance.
        """
        l = cls(robot.m_feats, robot.s_feats, robot.m_bounds, inv = inv, **kwargs)
        l.imodel.fmodel = fmodel
        return l

    @classmethod
    def from_robot_and_imodel(cls, robot, imodel):
        """ Create a learner from a robot instance and an inverse model

            :arg imodel:  an InverseModel instance.
        """
        l = cls(robot.Mfeats, robot.Sfeats, robot.Mbounds)
        l.imodel = imodel
        return l

    def __init__(self, Mfeats, Sfeats, Mbounds, fwd = 'LWLR', inv = 'L-BFGS-B', **kwargs):
        """

            :arg Mfeats:  the motor features (tuple of int)
            :arg Sfeats:  the sensory features (tuple of int, element-wise distinct from Mfeats)
            :arg Mbounds: the boundary describing legal motor values.
            :arg fwd:     string representing the kind of forward model to instanciate.
            :arg inv:     string representing the kind of inverse model to instanciate.
        """
        self.Mfeats  = Mfeats
        self.Sfeats  = Sfeats
        self.Mbounds = Mbounds
        fmodel = fwdclass[fwd](len(Mfeats), len(Sfeats), **kwargs)
        self.imodel  = invclass[inv].from_forward(fmodel, constraints = Mbounds, **kwargs)

    # Interface

    def add_xy(self, x, y):
        """ Add an motor order/sensory effect pair to the model

            :arg x:  an input (order) vector compatible with self.Mfeats.
            :arg y:  a output (effect) vector compatible with self.Sfeats.
        """
        self.imodel.add_xy(self._pre_x(x), self._pre_y(y))

    def infer_order(self, goal, **kwargs):
        """Infer an order in order to obtain an effect close to goal, given the
        observation data of the model.

        :arg goal:  a goal vector compatible with self.Sfeats
        :rtype:  same as goal (if list, tuple, or pandas.Series, else tuple)
        """
        x = self.imodel.infer_x(np.array(self._pre_y(goal)), **kwargs)[0]
        return self._post_x(x, goal)

    def predict_effect(self, order, **kwargs):
        """Predict the effect of a goal.

        :arg order:  an order vector compatible with self.Mfeats
        :rtype:  same as order (if list, tuple, or pandas.Series, else tuple)
        """
        y = self.imodel.fmodel.predict_y(np.array(self._pre_x(order)), **kwargs)
        return self._post_y(y, order)

    # Pre and post treatment

    def _pre_x(self, x):
        """Perform test on x and transform it into a tuple"""
        if type(x).__name__ == 'Series':
            assert set(x.index).issuperset(set(self.Mfeats)), ("Error :"
                    + "expected x.index = %s to be included in self.Mfeats = %s"
                    % (x.index, self.Mfeats))
            return tuple(x.reindex(self.Mfeats))
        else:
            assert len(x) == len(self.Mfeats)
            return tuple(x)

    def _pre_y(self, y):
        """Perform test on y and transform it into a tuple"""
        if type(y).__name__ == 'Series':
            assert set(y.index).issuperset(set(self.Sfeats))
            return tuple(y.reindex(self.Sfeats))
        else:
            assert len(y) == len(self.Sfeats)
            return tuple(y)

    def _post_x(self, result, input_object):
        """Return the result in the same format as the input."""
        if type(input_object).__name__ == 'Series':
            return pandas.Series(result, index = self.Mfeats)
        else:
            return tuple(result)

    def _post_y(self, result, input_object):
        """Return the result in the same format as the input."""
        if type(input_object).__name__ == 'Series':
            return pandas.Series(result, index = self.Sfeats)
        else:
            return tuple(result)
