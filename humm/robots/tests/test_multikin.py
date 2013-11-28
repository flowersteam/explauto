import testenv
import robots

# Robot
def test_kin():
    """Test if KinematicArm2D instanciate properly"""
    check = True

    arm = robots.KinematicArm2D(dim = 6)
    arm = robots.KinematicArm2D(dim = 1, lengths = 1.0)
    check = check and arm.dim == 1
    arm = robots.KinematicArm2D(dim = 6, lengths = 1.0, limits = (-1.0, 1.0))
    check = check and arm.m_feats == tuple(range(-6, 0))
    arm = robots.KinematicArm2D(dim = 6, lengths = 6*(1.0,), limits = 6*((-20.0, 20.0),))
    check = check and arm.m_bounds == 6*((-20.0, 20.0),)
    arm = robots.KinematicArm2D(dim = 12, m_feats = range(12))
    check = check and arm.dim == 12
    arm = robots.KinematicArm2D(dim = 12, s_feats = range(2))

    return check

tests = [test_kin]

if __name__ == "__main__":
    print("\033[1m%s\033[0m" % (__file__,))
    for t in tests:
        print('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__))
