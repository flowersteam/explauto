from .optimize import OptimizedInverseModel

# Number of independent optimization routine to run competively.
N_SAMPLE = 3
# Number of different infered answer to return. Should be <= N_SAMPLE.
N_RESULT = 1
assert N_RESULT < N_SAMPLE

from . import cma

class CMAESInverseModel(OptimizedInverseModel):
    """
    An inverse model class using CMA-ES optimization routine,
    on an error function computed from the forward model.
    """
        
    name = 'CMAES'
    
    desc = 'CMA-ES, Covariance Matrix Adaptation Evolution Strategy'

    
    def _setuplimits(self, constraints):
        OptimizedInverseModel._setuplimits(self, constraints)
        self.upper = tuple(c[1] for c in self.constraints)
        self.lower = tuple(c[0] for c in self.constraints)
        
    def infer_x(self, y, sigma = 1.0, tolerance = 1e-2):
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
            #self.lower = xg-sigma*5
            #self.upper = xg+sigma*5
            res = cma.fmin(self._error, xg, 1.0, 
                           bounds = [self.lower, self.upper], 
                           verb_log=0,
                           verb_disp=False,
                           #ftarget = tolerance/2,
                           maxfevals = 400)
            result.append((res[1], i, res[0]))

#        return [xi for fi, i, xi in sorted(result)[:min(N_RESULT, len(result))]]
        return [xi for fi, i, xi in sorted(result)]
