from collections import namedtuple

Settings = namedtuple('Settings', ('environment',
                                   'environment_config',
                                   'babbling_mode',
                                   'interest_model',
                                   'interest_model_config',
                                   'sensorimotor_model',
                                   'sensorimotor_model_config',
                                   'n_bootstrap'))


def make_settings(environment,
                  babbling_mode,
                  interest_model, sensorimotor_model,
                  environment_config='default',
                  interest_model_config='default',
                  sensorimotor_model_config='default',
                  n_bootstrap=0):

    return Settings(environment, environment_config,
                    babbling_mode,
                    interest_model, interest_model_config,
                    sensorimotor_model, sensorimotor_model_config, n_bootstrap)

from .experiment import Experiment
from .pool import ExperimentPool
