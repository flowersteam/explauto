# -*- coding: utf-8 -*-

# import sys, os
# 
# import numpy as np
# 
# import inverse
# 
# class JacobianInverseModel(inverse.InverseModel):
#     """Jacobian Inverse Model. #FIXME DRAFT DRAFT DRAFT NOTCHECKED BUGGY"""
#     
#     name = "Jacobian"
#     desc = 'Jacobian'
# 
#     
#     def infer_x(self, y_desired, **kwargs):
#         """Provide an inversion of yq in the input space
#         
#         @param yq  an array of float of length dim_y
#         """
#         sigma = kwargs.get('sigma', self.sigma)
#         k     = kwargs.get('k', self.k)
#         
#         xq = _guess_x(y_desired, k = k, sigma = sigma, **kwargs)[0]    
#             
#         w, index = self._weights(xq, k, sigma, **kwargs)
#         
#         Xq  = np.array(np.append([1.0], xq), ndmin = 2)
#         X   = np.array([self.dataset.data[0][i] for i in index])
#         Y    = np.array([self.dataset.data[1][k] for k in index])
#         
#         W   = np.diag(w)
#         WX  = np.dot(W, X)
#         WXT = WX.T
#             
#         B   = np.dot(np.linalg.pinv(np.dot(WXT, WX)),WXT)
#         
#         M = np.dot(B, np.dot(W, Y))
#         
#         inv = np.linalg.pinv(M[1:,:])
#         return np.dot(yq-M[0,:], inv)    
