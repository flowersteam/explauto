import numpy as np

from models.crossval    import CrossValidation
from models.forward.lwr import LWLRForwardModel

model = LWLRForwardModel(1, 1, sigma = 1.0)

for i in np.arange(-100, 100, 0.1):
    model.add_xy([i], [i*i])

for sigma in [0.0, 0.0001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 1000.0]:
    model.sigma = sigma
    cv = CrossValidation.from_model(model)
    print "%5.2f\t%7.4f" % (sigma, cv.k_folds(10))