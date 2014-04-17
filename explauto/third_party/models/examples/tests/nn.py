import random
import numpy as np
np.set_printoptions(precision = 2)

import models.forward.nn as nnmodel

def add(x):
    return (x[0]+x[1],)
# Creating model for function add
model = nnmodel.NNForwardModel(2, 1)
# Adding 5 random samples
model.add_xy(( 1.0, 1.0), (2.0,))
model.add_xy(( 0.0, 2.0), (2.0,))
model.add_xy((-1.0, 1.0), (0.0,))

# Testing on a random order
x = tuple(random.uniform(-10.0, 10.0) for _ in range(2))
y_predicted = model.predict_y(x)
y_actual    = np.array(add(x))
print y_predicted, " should be roughly ", y_actual
