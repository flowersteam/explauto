"""
Inverse model benchmark.
Inverse model performance is tested in function of the forward model, 
the inverse and forward parameters and the robots.
"""

import random
random.seed(30)

import robots
from models.learner import Learner
from models.plots.ercurve import ErrorCurve

# Robot
arm = robots.KinematicArm2D(dimM = 6)

# Learner
invlearners = ([Learner.from_robot(arm, fwd = 'WNN',   inv = 'BFGS',   k = 5, sigma = 5.0)]
			#+  [Learner.from_robot(arm, fwd = 'WNN',   inv = 'BFGS',          sigma = 10.0)]
			#+  [Learner.from_robot(arm, fwd = 'AvgNN', inv = 'BFGS',   k = 5, sigma = 5.0)]
			+  [Learner.from_robot(arm, fwd = 'ES-LWLR', inv = 'BFGS', k = 20)]
			)

# Plot
curve = ErrorCurve(arm, invlearners, side = 'inverse', trials = 1000, tests = 250, uniformity = 'sensor')
curve.plot()
curve.show()
