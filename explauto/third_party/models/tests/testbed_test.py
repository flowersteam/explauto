import testenv

try:
    import robots
    robotsimported = True
except Exception:
    print('%s %s' % ('\033[1;33mSKIP\033[0m', 'robots package import failed. tests not run.'))
    robotsimported = False

def test_testbed_coverage():
    """Typical code usage of testbed."""

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
    tb0.train_motor(10)
    for tb in testbeds:
        tb.fmodel.dataset = tb0.fmodel.dataset

    # Sharing testcases
    tb0.uniform_motor(10)
    for tb in testbeds:
        tb.testcases = tb0.testcases

    # Testing
    for tb in testbeds:
        errors = tb.run_forward()
        avg, std = tb.avg_std(errors)
        fwdname = tb.fmodel.__class__.__name__

    return True

tests = [test_testbed_coverage]

if __name__ == "__main__" and robotsimported:
    print("\033[1m%s\033[0m" % (__file__,))
    for t in tests:
        print('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__))
