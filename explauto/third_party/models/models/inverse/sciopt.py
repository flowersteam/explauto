
import numpy as np
import scipy.optimize

from .optimize import OptimizedInverseModel

# Number of independent optimization routine to run competively.
N_SAMPLE = 3
# Number of different infered answer to return. Should be <= N_SAMPLE.
N_RESULT = 1

from ..forward.lwr import LWLRForwardModel

class ScipyInverseModel(OptimizedInverseModel):
    """
    An inverse model class using optimization class of scipy (e.g. gradient descent, BFGS),
    on an error function computed from the forward model.
    """

    def __init__(self, dim_x, dim_y, constraints = (), algo = 'L-BFGS-B', **kwargs):
        raise NotImplementedError

    def infer_x(self, y):
        """Infer probable x from input y
        @param y  the desired output for infered x.
        @return   a list of probable x
        """
        OptimizedInverseModel.infer_x(self, y)
        if self.fmodel.size() == 0:
            return self._random_x()

        x_guesses = self._guess_x(y)

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

#        return [xi for fi, i, xi in sorted(result)[:min(N_RESULT, len(result))]]
        return [self._enforce_bounds(xi) for fi, i, xi in sorted(result)]

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

    def __init__(self, dim_x, dim_y, constraints = (),
                 maxiter =  500,
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
        OptimizedInverseModel.__init__(self, dim_x, dim_y, constraints = constraints, **kwargs)
        self.bounds = constraints
        self.conf   = {'maxiter' : maxiter,
                       'ftol'    : ftol,
                       'gtol'    : gtol,
                       'maxcor'  : maxcor,
                       'disp'    : disp,
                      }

class COBYLAInverseModel(ScipyInverseModel):
    """Class that takes specialized COBYLA options"""

    name = 'COBYLA'
    desc = 'COBYLA'
    algo = 'COBYLA'

    def __init__(self, dim_x, dim_y, constraints = (),
                 maxiter =  500,
                 rhoend  = 1e-3,
                 rhobeg  =  1.0,
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
