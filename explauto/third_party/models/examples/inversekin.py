# Create a regression problem for a non-linear, 6 DOFs kinematics arm in 3D space.

import sys
import random
import math
import numpy as np
import pandas

import toolbox

import models.samples as samples
import models.inverse.sciopt as iopt
import models.inverse.cmamodel as cmaes

random.seed(12345678)
#np.random.seed(9876543)

if __name__ == "__main__":
    ntests   = 100
    nsamples = 10000

    # Create a dataset of observations for an inverse kinematic problem.
    dset, f, bounds = samples.sample_6dof3d(nsamples)
    dset_y = list(dset.iter_y())

    # Create models
    model_lbfgsb         = iopt.BFGSInverseModel.from_dataset(dset, constraints = bounds)
    model_lbfgsb_exhaust = iopt.BFGSInverseModel.from_dataset(dset, constraints = bounds)
    model_lbfgsb_exhaust._guess_x_improved = model_lbfgsb_exhaust._guess_x
    model_cmaes          = cmaes.CMAESInverseModel.from_dataset(dset, constraints = bounds)
    model_cmaes_exhaust  = cmaes.CMAESInverseModel.from_dataset(dset, constraints = bounds)
    model_cmaes_exhaust._guess_x_improved = model_cmaes_exhaust._guess_x

    error_lbfgsb, error_lbfgsb_exh, error_cmaes, error_cmaes_exh, error_nn, error_random = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(ntests):
        print "%i/%i\r" % (i+1, ntests),
        sys.stdout.flush()
        goal = (random.uniform(0.0, -10.0), random.uniform(-20.0, 20.0), 0.0)
        sol_lbfgsb     = model_lbfgsb.infer_x(goal)[0]
        sol_lbfgsb_exh = model_lbfgsb_exhaust.infer_x(goal)
        sol_cmaes      = model_cmaes.infer_x(goal, sigma = 0.1, tolerance = 0.1)[0]
        sol_cmaes_exh  = model_cmaes.infer_x(goal, sigma = 0.1, tolerance = 0.1)
        dist, index = dset.nn_y(goal, 1)
        sol_nn = dset.get_x(index[0])
        # Updating errors
        error_lbfgsb     += toolbox.dist(goal, f(sol_lbfgsb))
        error_lbfgsb_exh += min([toolbox.dist(goal, f(sol_lbfgsb_i)) for sol_lbfgsb_i in sol_lbfgsb_exh])
        error_cmaes      += toolbox.dist(goal, f(sol_cmaes))
        error_cmaes_exh  += min([toolbox.dist(goal, f(sol_cmaes_i)) for sol_cmaes_i in sol_cmaes_exh])
        error_nn         += toolbox.dist(goal, f(sol_nn))
        error_random     += toolbox.dist(goal, random.choice(dset_y))

    print "Done %i tests on a %i samples dataset of a 6 DOFs kinematic arm." % (ntests, nsamples)
    print "L-BFGS-B               : %2.2f" % (error_lbfgsb/ntests)
    print "L-BFGS-B (exhaustive)  : %2.2f" % (error_lbfgsb_exh/ntests)
    print "CMAES                  : %2.2f" % (error_cmaes/ntests)
    print "CMAES (exhaustive)     : %2.2f" % (error_cmaes_exh/ntests)
    print "NN                     : %2.2f" % (error_nn/ntests)
    print "Random                 : %2.2f" % (error_random/ntests)
