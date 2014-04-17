import random
random.seed(0)

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 200, precision = 2)

import robots
import models
import models.forward as forward
import models.testbed as testbed
import models.plots.heat as heat

# Robot
vm = robots.KinematicArm2D(dim = 3)

# Forward Model
fwd = forward.WeightedNNForwardModel.from_robot(vm)

# Testbed
tb = testbed.Testbed(vm, fwd)

# Training model and creating tests
cases = tb.uniform_motor(300)
for x, y in cases:
    fwd.add_xy(x, y)
tb.reset_testcases()
tb.uniform_sensor(500)

# Heatmap
hmf = heat.HotTestbed(tb)
hmf.plot_fwd(res = 50)
plt.show()
