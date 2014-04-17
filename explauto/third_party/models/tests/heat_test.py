import testenv

def test_heatmap_coverage():
    """Execute Heatmap code to see if most of the things run without crashing"""
    import random
    random.seed(0)

    import numpy as np
    import matplotlib.pyplot as plt

    np.set_printoptions(linewidth = 200, precision = 2)

    import models
    import robots
    import models.forward as forward
    import models.testbed as testbed
    import models.plots.heat as heat

    # Robot
    #vm = robots.VowelModel()
    vm = robots.KinematicArm2D(dim = 3)
    # Forward Model
    fwd = forward.WeightedNNForwardModel.from_robot(vm)
    # Testbed
    tb = testbed.Testbed(vm, fwd)
    # Training model and creating tests
    cases = tb.uniform_motor(10)
    for x, y in cases:
        fwd.add_xy(x, y)
    tb.reset_testcases()
    tb.uniform_sensor(10)

    # Heatmap
    hmf = heat.HotTestbed(tb)
    hmf.plot_fwd(res = 5)
    return True


tests = [test_heatmap_coverage]

if __name__ == "__main__":
    print("\033[1m%s\033[0m" % (__file__,))
    for t in tests:
        print('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__))