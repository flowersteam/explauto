import numpy as np
import numpy.linalg
import gmr as fabisch_gmr
import cma


from numpy.linalg import inv, eig
from numpy import ix_, array, inf, sqrt, linspace, zeros, arctan2, matrix, pi
from .gaussian import Gaussian


class GMR(fabisch_gmr.gmm.GMM): 
    def __init__(self, n_components):
        fabisch_gmr.gmm.GMM.__init__(self, n_components)

    def probability(self, value):
        """Compute the probability of x knowing y 
        
        Parameters
        ----------
        value : array, shape(n_samples, n_features)
            Y
    
        Returns
        -------
        proba : float,
            X
        """
        
        proba = 0.
        pmc = zip(self.priors, self.means, self.covariances)
        for p,m,c in pmc:
            proba += p * Gaussian(m.reshape(-1,), c).normal(value.reshape(-1,))
        return proba
                   
    def regression(self, in_dims, out_dims, value, regression_method="slse", **kwargs):
        """ Compute a prediction y knowing a value x 
        
        Parameters
        ----------
        in_dims : array, shape (n_input_features,)
            Indices of dimensions of the input
        
        out_dims : array, shape (n_output_features,)
            Indices of dimensions of the output
        
        value : array, shape (n_input_features,)
            Value of the input
        
        regression_method : string
        
        Returns
        -------
        y : array, shape(n_output_features,)
            Value of the output
        
        """
        if regression_method == "lse":
            return self.predict(in_dims, value)[0]
        elif regression_method == "slse":
            return self.regression_slse(in_dims, out_dims, value)
        elif regression_method == "optimization":
            return self.regression_optimization(in_dims, out_dims, value, **kwargs)
        elif regression_method == "stochastic_sampling":
            return self.regression_stochastic_sampling(in_dims, out_dims, value)
        else: 
            raise NotImplementedError
            
        
    def regression_slse(self, in_dims, out_dims, value):
        """ 
        SLSE is the Single component LSE and computes the most probably value of the most important gaussian 
        knowing an input value.
        """
        conditional_gmm = self.condition(in_dims, value)
        max_prior = np.max(conditional_gmm.priors)
        max_ind = np.where(conditional_gmm.priors == max_prior)[0][0]
        return conditional_gmm.means[max_ind]  
     
    def regression_optimization(self, in_dims, out_dims, value, x_guess,
                                optimization_parameters={"sigma":0.1, "maxfevals":200, "bounds":None},
                                optimization_method="CMAES"):
        """
        This sampling is the obtained by the minimization of the error x knowing y by using an optimization method 
        as CMAES or BFGS
        """
        conditional_gmm = self.condition(in_dims, value)
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
        return conditional_gmm.sample(n_samples=1)[0]
   
 
