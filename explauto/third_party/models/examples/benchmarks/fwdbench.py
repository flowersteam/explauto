"""Benchmarking a simple 3 DOF, 2D Kinematic Arm in function of timesteps."""
import random
random.seed(0)

import robots
import models
from models.learner import Learner, fwdclass
from models.plots.ercurve import ErrorCurve

#models.disable_cmodels()

# Robot
arm = robots.KinematicArm2D(dim = 3)

#fwdlearners = [Learner.from_robot(arm, fwd = fwd, inv = 'NN', k = 5, sigma = 5.0) for fwd in fwdclass]
fwdlearners = []

#fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'NN', k = 4))
#fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'NN', k = 7))
fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'NN', k = 3))
fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'NN', k = 7))
fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'NN', k = 10))
fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'NN', k = 20))

curve = ErrorCurve(arm, fwdlearners, side = 'forward', trials = 2000, tests = 200, uniformity = 'motor', test_period = 5, test_pace = 1.5)
curve.plot()
curve.show()
