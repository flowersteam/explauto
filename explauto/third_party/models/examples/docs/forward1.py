import random
import math
import numpy as np
np.set_printoptions(precision = 2)

import models.samples as samples
import models.forward.lwr as lwr

nsamples = 1000

# Creating a sample dataset out of a 6DOF robotic arm simulation.
# The dataset dimension is 6 by 3. f is the generative function.
dset, f, bounds = samples.sample_6dof3d(nsamples)
# Creating the model from the dataset.
model = lwr.LWLRForwardModel.from_dataset(dset, sigma = 1.0)
    
# Testing on order [ 0. 0. 0. 0. 0. 0.]
x = (0, 0, 0, 0, 0, 0)
y_predicted = model.predict_y(x)
y_actual    = np.array(f(x))
print y_predicted, " should be roughly ", y_actual

# Testing on a random order
x = tuple(random.uniform(-math.pi, math.pi) for _ in range(6))
y_predicted = model.predict_y(x)
y_actual    = np.array(f(x))
print y_predicted, " should be roughly ", y_actual
