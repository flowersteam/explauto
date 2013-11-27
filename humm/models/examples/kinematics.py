# Create a regression problem for a non-linear, 6 DOFs kinematics arm in 3D space.

import random
import math
import numpy as np

import models.samples as samples
import models.forward.lwr as lwr

if __name__ == "__main__":
    ntests   = 100
    nsamples = 1000

    dset, f, bounds = samples.sample_6dof3d(nsamples)
    model = lwr.LWLRForwardModel.from_dataset(dset, sigma = 1.0)
    d_lwr = 0.0
    d_nn  = 0.0
    goals = []
    for i in range(ntests):
        x = tuple(random.uniform(-math.pi, math.pi) for _ in range(6))
        y_p = model.predict_y(x)
        dist, index = model.dataset.nn_x(x, 1)
        y_nn = model.dataset.get_y(index[0])
        y_r = f(x)
        goals.append(y_r)
        d_lwr += math.sqrt(sum((y_i-y_ri)**2 for y_i, y_ri in zip(y_p, y_r)))
        d_nn  += math.sqrt(sum((y_i-y_ri)**2 for y_i, y_ri in zip(y_nn, y_r)))

    std = math.sqrt(sum(yi**2 for yi in np.std(goals, axis=0)))

    print "Done %i tests on a %i samples dataset of a 6 DOFs kinematic arm." % (ntests, nsamples)
    print "Result : lwr:%.2f/nn:%.2f/random:%.2f error per sample." % (d_lwr/ntests, d_nn/ntests,std)
