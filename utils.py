import numpy as np

def rand_bounds(bounds):
    widths=bounds[1,:] - bounds[0,:]
    return (widths * np.random.rand(1, bounds.shape[1]) + bounds[0,:]).reshape(-1,1)

def bounds_min_max(v, mins, maxs):
    res = np.minimum(v, maxs)
    res = np.maximum(v, mins)    
    return res
