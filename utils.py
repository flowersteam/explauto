import numpy as np
from scipy import stats

def rand_bounds(bounds):
    widths=bounds[1,:] - bounds[0,:]
    return (widths * np.random.rand(1, bounds.shape[1]) + bounds[0,:]).reshape(-1,1)

def bounds_min_max(v, mins, maxs):
    res = np.minimum(v, maxs)
    res = np.maximum(res, mins)    
    return res


def discrete_random_draw(data,nb=1):
    ''' Code from Steve Nguyen'''
    data=np.array(data)
    if not data.any():
        data = np.ones_like(data)
    data=data/data.sum()
    xk = np.arange(len(data))
    custm = stats.rv_discrete(name='custm', values=(xk, data))
    return custm.rvs(size=nb)


#class IncrementalVariance(object):
    #def __init__(self):
        #self.new_old_ratio = 0.1
        #pass
    #def update(self, x):

#def weighted_incremental_variance(dataWeightPairs):
    #sumweight = 0
    #mean = 0
    #M2 = 0
 
    #for x, weight in dataWeightPairs:  # Alternatively "for x, weight in zip(data, weights):"
        #temp = weight + sumweight
        #delta = x - mean
        #R = delta * weight / temp
        #mean = mean + R
        #M2 = M2 + sumweight * delta * R  # Alternatively M2 = M2 + weight * delta * (x-mean)
        #sumweight = temp
 
    #variance_n = M2/sumweight
    #variance = variance_n * len(dataWeightPairs)/(len(dataWeightPairs) - 1)
