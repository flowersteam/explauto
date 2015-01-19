# from config import environment, configurations, testcases
from .diva import DivaEnvironment, DivaSynth
from .config import default_config, vowel_config, low_config, full_config


environment = DivaEnvironment
configurations = {'default': default_config,
                  'vowel_config': vowel_config,
                  'low_config': low_config,
                  'full_config': full_config
                 }

def testcases(config_str, n_samples=100):
    raise NotImplementedError('No testcases available for the diva environment')
