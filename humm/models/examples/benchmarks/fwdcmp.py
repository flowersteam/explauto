"""A straightforward example of model benchmarking"""

import robots
import models.learner as learner
import models.testbed as testbed

import random
random.seed(0)

# Robot
arm = robots.KinematicArm2D(dim = 3)

# Instanciating testbed
fwdlearners = [learner.Learner.from_robot(arm, fwd = fwd, inv = 'NN') for fwd in learner.fwdclass]
testbeds    = [testbed.Testbed.from_learner(arm, learnr) for learnr in fwdlearners]

# Sharing training
tb0 = testbeds[0]
tb0.train_motor(100)
for tb in testbeds:
    tb.fmodel.dataset = tb0.fmodel.dataset

# Sharing testcases
tb0.uniform_motor(100)
for tb in testbeds:
    tb.testcases = tb0.testcases

# Testing
for tb in testbeds:
    errors = tb.run_forward()
    avg, std = tb.avg_std(errors)
    fwdname = tb.fmodel.__class__.__name__
    print "%s%s: %5.2f" % (fwdname, (30-len(fwdname))*" ", avg)