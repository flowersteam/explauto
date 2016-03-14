import numpy as np
import numpy.linalg
import gmr as fabisch_gmr


from numpy.linalg import inv, eig
from numpy import ix_, array, inf, sqrt, linspace, zeros, arctan2, matrix, pi
from .gaussian import Gaussian
from .models import cma


class GMR(fabisch_gmr.gmm.GMM): 
    def __init__(self, n_components):
        fabisch_gmr.gmm.GMM.__init__(self, n_components)

    def probability(self, value):
        """
        return the probability of x knowing y 
        """
        proba = 0.
        pmc = zip(self.priors, self.means, self.covariances)
        for p,m,c in pmc:
            proba += p * Gaussian(m.reshape(-1,), c).normal(value.reshape(-1,))
        return p
                   
    def regression(self, regression_method="slse", in_dims, out_dims, value, **kwargs):
        if regression_method == "lse":
            return self.regression_lse(in_dims, out_dims, value)
        elif regression_method == "slse":
            return self.regression_slse(in_dims, out_dims, value)
        elif regression_method == "optimization":
            return self.regression_optimization(in_dims, out_dims, value, **kwargs)
        elif regression_method == "stochastic_sampling":
            return self.regression_stochastic_sampling(in_dims, out_dims, value)
        else: 
            raise NotImplementedError
            
    def regression_lse(self, in_dims, out_dims, value):
        """
        LSE is the least square estimate that computes the most probably value knowing an input value.
        It is the computation of the weighted mean of a value for each gaussian of the GMM
        """
        conditional_gmm = self.condition(in_dims, value)
        assert(len(conditional_gmm)==out_dims)
        return conditional_gmm.weights.dot(conditional_gmm.means)
        
    def regression_slse(self, in_dims, out_dims, value):
        """ 
        SLSE is the Single component LSE and computes the most probably value of the most important gaussian 
        knowing an input value.
        """
        conditional_gmm = self.condition(in_dims, value)
        assert(len(conditional_gmm)==out_dims)
        max_weight = np.max(conditional_gmm.weights)
        max_ind = np.where(conditional_gmm.weights == max_weight)[0][0]
        return conditional_gmm.weights.means[max_ind]  
     
    def regression_optimization(self, in_dims, out_dims, value, optimization_method="CMAES", 
                                optimization_parameters={"sigma":0.1, "maxfevals":200, "bounds":None},
                                x_guess):
        """
        This sampling is the obtained by the minimization of the error x knowing y by using an optimization method 
        as CMAES or BFGS
        """
        conditional_gmm = self.condition(in_dims, value)
        assert(len(conditional_gmm)==out_dims)
        f = lambda x: - conditional_gmm.probablity(x)
        
        if optimization_method=="CMAES":
            res = cma.fmin(f, x_guess, optimization_parameters["sigma"], 
                       options={'bounds':optimization_parameters["bounds"],
                       'verb_log':0,
                       'verb_disp':False,
                       'maxfevals':optimization_parameters["maxfevals"]})
        
        elif optimization_method=="BFGS": #TODO
            raise NotImplementedError
        
        else:
            raise NotImplementedError
        # .......
        
        return res[0] #index to check, it might be 1
   
    def regression_stochastic_sampling(self, in_dims, out_dims, value):
        """
        This method returns a random value according to the conditional probability
        """   
        conditional_gmm = self.condition(in_dims, value)
        assert(len(conditional_gmm)==out_dims)
        return conditional_gmm.sample(n_samples=1)[0]
   
 
