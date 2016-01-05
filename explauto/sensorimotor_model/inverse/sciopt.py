
import numpy as np
import scipy.optimize

from .optimize import OptimizedInverseModel


class ScipyInverseModel(OptimizedInverseModel):
    """
    An inverse model class using optimization class of scipy (e.g. gradient descent, BFGS),
    on an error function computed from the forward model.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

    def infer_x(self, y):
        """Infer probable x from input y
        @param y  the desired output for infered x.
        @return   a list of probable x
        """
        OptimizedInverseModel.infer_x(self, y)
        if self.fmodel.size() == 0:
            return self._random_x()

        x_guesses = [self._guess_x_simple(y)[0]]
        result = []
        for i, xg in enumerate(x_guesses):
            res = scipy.optimize.minimize(self._error, xg,
                                          args        = (),
                                          method      = self.algo,
                                          bounds      = self.constraints,
                                          options     = self.conf
                                         )



            d = self._error(res.x)
            result.append((d, i, res.x))
        return [self._enforce_bounds(xi) for fi, i, xi in sorted(result)]
    
    def infer_dm(self, m, s, ds):
        """Infer probable output from input x, y
        """
        OptimizedInverseModel.infer_dm(self, ds)
        if len(self.fmodel.dataset) == 0:
            return [[0.0]*self.dim_out]
        else:
            _, index = self.fmodel.dataset.nn_dims(m, np.hstack((s, ds)), range(len(m)), range(self.dim_x, self.dim_x + self.dim_y), k=1)
            guesses = [self.fmodel.dataset.get_dims(index[0], dims=range(len(m), self.dim_x))]
            result = []
                    
            for g in guesses:
                res = scipy.optimize.minimize(lambda dm:self._error_dm(m, dm, s), g,
                                              args        = (),
                                              method      = self.algo,
                                              options     = self.conf
                                             )
    
    
    
                d = self._error_dm(m, res.x, s)
                result.append((d, res.x))
            return [xi for fi, xi in sorted(result)][0]
            

    def _enforce_bounds(self, x):
        """"Enforce the bounds on x if only infinitesimal violations occurs"""
        assert len(x) == len(self.bounds)
        x_enforced = []
        for x_i, (lb, ub) in zip(x, self.bounds):
            if x_i < lb:
                if x_i > lb - (ub-lb)/1e10:
                    x_enforced.append(lb)
                else:
                    x_enforced.append(x_i)
            elif x_i > ub:
                if x_i < ub + (ub-lb)/1e10:
                    x_enforced.append(ub)
                else:
                    x_enforced.append(x_i)
            else:
                x_enforced.append(x_i)
        return np.array(x_enforced)


class BFGSInverseModel(ScipyInverseModel):
    """Class that takes specialized L-BFGS-B options"""

    name = 'L-BFGS-B'
    desc = 'L-BFGS-B'
    algo = 'L-BFGS-B'

    def __init__(self, dim_x=None, dim_y=None, fmodel=None, constraints = (),
                 maxfun =  50,
                 ftol    = 1e-5,
                 gtol    = 1e-3,
                 maxcor  =   10,
                 disp    = False,
                 **kwargs):
        """
        * L-BFGS-B options (from scipy doc):
        ftol : float
            The iteration stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        gtol : float
            The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
            <= gtol`` where ``pg_i`` is the i-th component of the
            projected gradient.
        maxcor : int
            The maximum number of variable metric corrections used to
            define the limited memory matrix. (The limited memory BFGS
            method does not store the full hessian but uses this many terms
            in an approximation to it.)
        maxiter : int
            Maximum number of function evaluations.
        """
        OptimizedInverseModel.__init__(self, dim_x, dim_y, fmodel=fmodel, constraints = constraints, **kwargs)
        self.bounds = constraints
        self.conf   = {'maxfun' : maxfun,
                       'ftol'    : ftol,
                       'gtol'    : gtol,
                       'maxcor'  : maxcor,
                       'disp'    : disp,
                      }

    def _guess_x(self, y_desired, **kwargs):
        _, indexes = self.fmodel.dataset.nn_y(y_desired, k=1)
        return [self.fmodel.dataset.get_x(indexes[0])]


class COBYLAInverseModel(ScipyInverseModel):
    """Class that takes specialized COBYLA options"""

    name = 'COBYLA'
    desc = 'COBYLA'
    algo = 'COBYLA'

    def __init__(self, dim_x=None, dim_y=None, constraints = (),
                 maxiter =  50,
                 rhoend  = 1e-2,
                 rhobeg  =  0.05,
                 disp    = False,
                 **kwargs):
        """
        * COBYLA options (from scipy doc):
        COBYLA options:
        rhobeg : float
            Reasonable initial changes to the variables.
        rhoend : float
            Final accuracy in the optimization (not precisely guaranteed).
            This is a lower bound on the size of the trust region.
        maxfev : int
            Maximum number of function evaluations.
        """
        OptimizedInverseModel.__init__(self, dim_x, dim_y, constraints = constraints, **kwargs)
        self.bounds = constraints
        self.conf   = {'maxiter': maxiter,
                       'tol'    : rhoend,
                       'rhobeg' : rhobeg,
                       'disp'    : disp,
                      }
