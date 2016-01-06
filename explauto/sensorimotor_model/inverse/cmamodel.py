
from .optimize import OptimizedInverseModel
from . import cma
    

class CMAESInverseModel(OptimizedInverseModel):
    """
    An inverse model class using CMA-ES optimization routine,
    on an error function computed from the forward model.
    """
        
    name = 'CMAES'
    desc = 'CMA-ES, Covariance Matrix Adaptation Evolution Strategy'
    
    def __init__(self, dim_x=None, dim_y=None, fmodel=None, cmaes_sigma=0.05, maxfevals=20, **kwargs):
        self.cmaes_sigma = cmaes_sigma
        self.maxfevals = maxfevals
        OptimizedInverseModel.__init__(self, dim_x, dim_y, fmodel=fmodel, **kwargs)
        
    def _setuplimits(self, constraints):
        OptimizedInverseModel._setuplimits(self, constraints)
        self.upper = list(c[1] for c in self.constraints)
        self.lower = list(c[0] for c in self.constraints)
        
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
        for xg in x_guesses:
            res = cma.fmin(self._error, xg, self.cmaes_sigma, 
                           options={'bounds':[self.lower, self.upper],
                           'verb_log':0,
                           'verb_disp':False,
                           'maxfevals':self.maxfevals})
            result.append((res[1], res[0]))
 
        return [xi for fi, xi in sorted(result)]
    
    def infer_dims(self, x, y, dims_x, dims_y, dims_out):
        """Infer probable output from input x, y
        """
        OptimizedInverseModel.infer_x(self, y)
        assert len(x) == len(dims_x)
        assert len(y) == len(dims_y)
        if len(self.fmodel.dataset) == 0:
            return [[0.0]*self.dim_out]
        else:
            _, index = self.fmodel.dataset.nn_dims(x, y, dims_x, dims_y, k=1)
            guesses = [self.fmodel.dataset.get_dims(index[0], dims_out)]
                    
            result = []
            for g in guesses:
                res = cma.fmin(lambda q:self._error_dims(q, dims_x, dims_y, dims_out), g, self.cmaes_sigma, 
                               options={'bounds':[self.lower, self.upper],
                               'verb_log':0,
                               'verb_disp':False,
                               'maxfevals':self.maxfevals})
                result.append((res[1], res[0]))
     
            return sorted(result)[0][1]
                
            
