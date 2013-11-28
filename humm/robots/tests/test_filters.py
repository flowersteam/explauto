import testenv
import robots

import treedict

# Robot
def test_filters0():
    """Test if KinematicArm2D instanciate properly with config"""
    check = True

    cfg = treedict.TreeDict()
    cfg.robotclass = 'robots.KinematicArm2D'
    bot = robots.build_robot(cfg)

    check = bot.s_feats == (0, 1)
    return check


def test_filters1():
    """Test if KinematicArm2D instanciate properly with filters"""
    check = True

    cfg = treedict.TreeDict()
    cfg.robotclass = 'robots.KinematicArm2D'
    cfg.filters.s_feats = (0,)
    bot = robots.build_robot(cfg)

    check = bot.s_feats == (0,)
    return check

# def test_filters2():
#     """Test if external classes instanciate properly with filters"""
#     check = True
#
#     cfg = treedict.TreeDict()
#     cfg.robotclass = 'surrogates.kinsim.KinSim'
#     cfg.filters.uniformzise = False
#     bot = robots.build_robot(cfg)
#
#     check = bot.s_feats == (0, 1, 2, 3)
#     return check



tests = [test_filters0,
         test_filters1,
         #test_filters2
        ]

if __name__ == "__main__":
    print("\033[1m%s\033[0m" % (__file__,))
    for t in tests:
        print('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
                         '\033[1;31mFAIL\033[0m', t.__doc__))
