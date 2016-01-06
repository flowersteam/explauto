
import numpy as np
 
import inverse

 
class JacobianInverseModel(inverse.InverseModel):
    """Jacobian Inverse Model."""
     
    name = "Jacobian"
    desc = 'Jacobian'
 
    def __init__(self, dim_x, dim_y, fmodel, **kwargs):
        inverse.InverseModel.from_forward(fmodel, **kwargs)
        self.fmodel = fmodel
        self.sigma = kwargs['sigma']
        self.k     = kwargs['k']
     
    def infer_x(self, y_desired, **kwargs):
        """Provide an inversion of yq in the input space
         
        @param yq  an array of float of length dim_y
        """
        
        sigma = kwargs.get('sigma', self.sigma)
        k     = kwargs.get('k', self.k)
         
        xq = self._guess_x(y_desired, k = k, sigma = sigma, **kwargs)[0]    
         
        dists, index = self.fmodel.dataset.nn_x(xq, k = k)

        w = self.fmodel._weights(dists, sigma*sigma)
        
        X   = np.array([self.fmodel.dataset.get_x_padded(i) for i in index])
        Y    = np.array([self.fmodel.dataset.get_y(i) for i in index])
         
        W   = np.diag(w)
        WX  = np.dot(W, X)
        WXT = WX.T
             
        B   = np.dot(np.linalg.pinv(np.dot(WXT, WX)),WXT)
        
        M = np.dot(B, np.dot(W, Y))
        
        _, idx = self.fmodel.dataset.nn_y(y_desired, k=1)        
        ynn = self.fmodel.dataset.get_y(idx[0])
        
        eps = 0.00001
        
        J = np.zeros((len(xq),len(y_desired)), dtype = np.float64)
        for i in range(len(xq)):
            xi = np.array(xq, dtype = np.float64)
            xi[i] = xi[i] + eps
            yi = np.dot(np.append(1.0, xi), M).ravel()
            J[i] = (yi - ynn) / eps
        J = np.transpose(J)
        Jinv = np.linalg.pinv(J)
        x = np.dot(Jinv, y_desired - ynn)
        
        return [x]
