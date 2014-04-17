import math
import random

from dataset import Dataset
from forward.lwr import LWLRForwardModel

class CrossValidation(object):
    """Compute the cross validation of a model and dataset"""
    
    @classmethod
    def from_dataset(cls, dataset, modelname = 'LWR', conf = None):
        modelclass = forward.LWLRForwardModel
        if modelname == 'NN':
            modelclass = forward.NNForwardModel
        return cls(dataset, modelclass, conf = conf)
        
    @classmethod
    def from_model(cls, model):
        return cls(model.dataset, model.__class__, conf = model.conf)
            
    def __init__(self, dataset, modelclass, conf = None):
        self.modelclass = modelclass
        self.dataset    = dataset
        self.modelconf  = conf or {}
        
    def _divide(self, n):
        """Divide the data into similar n partitions"""
        bins = [[] for i in range(n)]
        for datapoint in self.dataset.iter_xy():
            bins[random.randint(0, n-1)].append(datapoint)
        return bins
        
    def k_folds(self, n):
        """Return the n-folds cross-validation error"""
        bins = self._divide(n)
        error = 0.0
        for i in xrange(n):
            partialdata = sum((bin_j for j, bin_j in enumerate(bins) if i != j), [])
            partialds = Dataset.from_data(partialdata)
            partialmodel = self.modelclass.from_dataset(partialds, **self.modelconf)
            for x, y in bins[i]:
                e = math.sqrt(((y - partialmodel.predict_y(x))**2).sum())
                #print e
                error += e
                #print error
        
        return error/len(self.dataset)
