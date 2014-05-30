from numpy import array, pi

from ..simple_arm import make_arm_config
from .musical_arm import MusicalArm
from ... import ExplautoNoTestCasesError


x_min, y_min, x_max, y_max = 0., -1., 0.5, -0.1
pitches = [-0.6, 0.3, 0.5, 0.7, 0.9]
noises = [0.3] + [0.1] * (len(pitches) - 1)
config = make_arm_config(15, pi/3, array([-1., -1.]), array([1., 1.]), 1.5, 0.02)
config['pitches'] = pitches
config['pitch_noises'] = noises
config['box'] = x_min, y_min, x_max, y_max

environment = MusicalArm
configurations = {'default': config}


def testcases(config_str, n_samples=-1):
    return None
    # raise ExplautoNoTestCasesError("No testcases here")
