from numpy import array

from ...utils import rand_bounds
from .pendulum import PendulumEnvironment
from .config import test_config, x_min, x_max, v_min, v_max


environment = PendulumEnvironment
configurations = {'default': test_config}


def testcases(config_str, n_samples=100):
    if config_str == 'default':
        return rand_bounds(array([[x_min, v_min], [x_max, v_max]]), n_samples)
    else:
        ExplautoNoTestCasesError("Only works for default configuration")


            
