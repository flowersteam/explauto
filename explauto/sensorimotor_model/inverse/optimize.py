
import random

from . import inverse
from ..forward.lwr import LWLRForwardModel


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

    def __init__(self, dim_x=None, dim_y=None, fmodel=None, constraints=(), **kwargs):
        """Construst an inverse model from a dimensions and constraints set
        Default to a LWR model for the forward model.
        """      
        if fmodel:
            self.dim_x = fmodel.dim_x
            self.dim_y = fmodel.dim_y
            self.fmodel = fmodel
        else:
            self.dim_x = dim_x
            self.dim_y = dim_y
            self.fmodel = LWLRForwardModel(self.dim_x, self.dim_y, **kwargs) 
        inverse.InverseModel.__init__(self, self.dim_x, self.dim_y, **kwargs)  
        self._setuplimits(constraints)


    def _setuplimits(self, constraints):
        """Setup the limits for every initialiser."""
        self.constraints = tuple(constraints)
        assert len(self.constraints) == self.fmodel.dim_x
        self.y_desired = None

    def _random_x(self):
        return (tuple(random.uniform(b_min, b_max) for b_min, b_max in self.bounds),)

    def _error(self, x, y_dims=None):
        """Error function.
        Once self.y_desired has been defined, compute the error
        of input x using the forward model.
        """
        #print 'x', x
        y_pred = self.fmodel.predict_y(x, y_dims)
        err_v  = y_pred - self.y_desired
        error = sum(e*e for e in err_v)
        #print "OptimizedInverseModel y_desired:", self.y_desired, " x:", x, " ypred:", y_pred, " error:", error 
        return error
