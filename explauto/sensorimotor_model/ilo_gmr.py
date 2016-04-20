import numpy as np

from numpy import array, hstack

from ..exceptions import ExplautoBootstrapError
from .sensorimotor_model import SensorimotorModel
from explauto.models.dataset import Dataset

from .inverse.cmamodel import CMAESInverseModel as CMAES
from ..models.gmrinf import GMR
from sklearn.linear_model.sgd_fast import Regression

class IloGmr(SensorimotorModel, GMR):
    """GMR adapted to sensori motor models"""
    
    def __init__(self, conf, n_components=3): #choix methode dans init
        SensorimotorModel.__init__(self, conf)
        GMR.__init__(self, n_components)
        self.n_neighbors = max(100, (conf.ndims) ** 2)  # at least 100 neighbors
        self.n_neighbors = min(1000, self.n_neighbors)  # at most 1000 neighbors
        self.min_n_neighbors = 20  # otherwise raise ExplautoBootstrapError
        self.m_dims = conf.m_dims
        self.s_dims = conf.s_dims        
        
    def infer(self, in_dims, out_dims, value, method="direct", regression="SLSE"):
        if method=="direct":
            res = self.infer_direct(in_dims, out_dims, value, regression)
        elif method=="undirect":
            res = self.infer_undirect(in_dims, out_dims, value, regression)
        else:
            pass
        return res
    
    def update(self, m, s):
        self.dataset.add_xy(tuple(m), tuple(s))
        pass
    
    
    def infer_undirect(self, in_dims, out_dims, xq, regression = CMAES):

        cma = CMAES(in_dims, out_dims, fmodel=None, cmaes_sigma=0.05, maxfevals=20)
        
    def infer_direct(self,in_dims, out_dims, xq, regression = CMAES):
        pass
    
    
    #===================================================================================
    #BROUILLON
    def predict_direct(self, in_dims, out_dims, xq, method):
        """dire ce que fait la fct et d'ou elle vient"""

        #local_gmrinf = (3, local_gmm.priors_, local_gmm.means_, local_gmm.covars_)

        #conditional_gmm = local_gmrinf.condition(xq)
        
    #def infer_direct(self, in_dims, out_dims, xq, method == 1):
        """utilise directement methode sur modele inverse"""

        

        #manque optimisation
        
        
        #faire des fonctions pour chaque méthode i), ii), iii), iv) dans GMR
        #faire update et infer ici
        #toutes les méthodes (inverse direct et 4 étapes) + 4 méthodes i), ... fonctionnent avec GMM classique et GMM local donc les mettre dans classe GMR
        # => mettre les fonctions une classe GMM mais différent de GMMinf avec seulement ce qui est utile (appelé différent : GMR . GMR dans models
        #Mettre classe GMR dans models
        # sciopt et optimize à aller chercher pour implémenter méthode iii)
        #CMAES prend f(x) en entrée qu'on va définir avec GMM(x,y) ou |
        #Récupérer code de infer de CMAESInverseModel dans classe GMR après avoir construit le f(x) = GMM(x,y)
        #pour l'instant faire que avec CMAES et pas avec BFGS, etc.
        #dans inverse, sciopt, sciopt.optimize.minimize de infer_x , on pourra changer algo (plus tard)
        
        #GMR doit pouvoir faire des GMM conditionnelles (infer ou 
        #robuste inverse/direct
        #CMAES fmin utilisé dans méthode optimilisation (iii) )
        
        # Paramètre nombre de gaussienne de GMR
        
        #ILOGMR
        #il faut écrire une fonction par méthode ( 1) 4étapes ou 2) inverse directement. Je sépare comment pour faire inverse et direct ? 
        #GMR
        #j' ai encore du mal a voir différence entre ii) et iii) : 
        
        
        
        













