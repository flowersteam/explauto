"""Class shared by optimization-based inverse models."""

import random
import numpy as np

from ....toolbox import toolbox
from . import inverse

from ..forward.lwr import LWLRForwardModel
relax = 0.0 # no relaxation

class OptimizedInverseModel(inverse.InverseModel):

    @classmethod
    def from_dataset(cls, dataset, constraints = (), **kwargs):
        """Construct a optimized inverse model from an existing dataset.
        A LWLR forward model is constructed by default.
        """
        fm = LWLRForwardModel(dataset.dim_x, dataset.dim_y, **kwargs)
        fm.dataset = dataset
        im = cls.from_forward(fm, constraints = constraints, **kwargs)
        return im

    @classmethod
    def from_forward(cls, fmodel, constraints = (), **kwargs):
        """Construst an inverse model from a forward model and constraints.
        """
        im = cls(fmodel.dim_x, fmodel.dim_y, constraints = constraints, **kwargs)
        im.fmodel = fmodel
        return im

    def __init__(self, dim_x, dim_y, constraints = (), **kwargs):
        """Construst an inverse model from a dimensions and constraints set
        Default to a LWR model for the forward model.
        """
        inverse.InverseModel.__init__(self, dim_x, dim_y, **kwargs)
        self.fmodel = LWLRForwardModel(dim_x, dim_y, **kwargs)
        self._setuplimits(constraints)


    def _setuplimits(self, constraints):
        """Setup the limits for every initialiser."""
        self.constraints = tuple(constraints)
        assert len(self.constraints) == self.fmodel.dim_x
        self.y_desired = None


    # Will be useful for chained lenvs...

    # def _setuplimits(self, constraints):
    #     """Setup the limits for every initialiser."""
    #     self.limits      = tuple([float('inf'), float('-inf')] for _ in range(self.fmodel.dim_x - len(constraints)))
    #     self.constraints = tuple(constraints) + self.limits
    #
    #     self.y_desired = None

    # def _relax(x, factor):
    #     if x >= 0:
    #         return x*factor
    #     else:
    #         return x/factor

    # def add_x(self, x):
    #     """We do not store x, only update the constraints"""
    #     offset = len(self.constraints) - len(self.limits)
    #     for i, xi in enumerate(x):
    #         if i >= offset:
    #             il = i - offset
    #             limin, limax = self.limits[il]
    #             self.limits[il][0] = min(xi, limin)
    #             self.limits[il][0] = max(xi, limax)
    #             l = self.limits[il][1] - self.limits[il][0]
    #
    #             self.constraints[i][0] = self.limits[il][0] - relax*l
    #             self.constraints[i][1] = self.limits[il][1] + relax*l

    def _error(self, x):
        """Error function.
        Once self.y_desired has been defined, compute the error
        of input x using the forward model.
        """
        #print 'x', x
        y_pred = self.fmodel.predict_y(x)
        err_v  = y_pred - self.y_desired
        error = sum(e*e for e in err_v)
        return error
