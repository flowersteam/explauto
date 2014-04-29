"""Benchmarking a simple 3 DOF, 2D Kinematic Arm in function of timesteps."""
import random
random.seed(0)

import treedict
import robots
from models.learner import Learner, fwdclass
from models.plots.ercurve import ErrorCurve

# Robot
cfg = treedict.TreeDict()
cfg.dim     = 6
cfg.lengths = 1.0
cfg.limits  = (-360.0, 360.0)
arm = robots.KinematicArm2D(cfg)

fwdlearners = []
fwdlearners.append(Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'L-BFGS-B', k = 2*cfg.dim+1))

curve = ErrorCurve(arm, fwdlearners, side = 'forward', trials = 50000, tests = 200, uniformity = 'motor')
curve.plot()
curve.show()
