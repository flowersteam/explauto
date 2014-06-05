from numpy import array, pi

from ..simple_arm import make_arm_config, SimpleArmEnvironment
from ...utils.sound.pitch_tracker import estimate_key
from .music import MusicEnvironment

boxes2D_1 = [[[-1, 0.], [-1., 1.]], [[0., 1.], [-1., 0.]], [[0., 1.], [0., 1.]]]
wav_files = ['../data/piano_notes/a.wav', '../data/piano_notes/b.wav', '../data/piano_notes/c.wav']
sound_feature_min = 0.
sound_feature_max = 1.
noise = 0.02
arm_config = make_arm_config(15, pi/3, array([-1., -1.]), array([1., 1.]), 1.5, 0.02)
arm = SimpleArmEnvironment(**arm_config)

config = dict(base_environment=arm,
              boxes=boxes2D_1,
              sound_samples=wav_files,
              sound_analyser=estimate_key,
              analyser_noise=noise,
              sound_mins=sound_feature_min,
              sound_maxs=sound_feature_max,
              internal_play_and_record=True)


environment = MusicEnvironment
configurations = {'default': config}


def testcases(config_str, n_samples=-1):
    return None
    # raise ExplautoNoTestCasesError("No testcases here")
