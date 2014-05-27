from collections import namedtuple

Settings = namedtuple('Settings', ('environment',
                                   'environment_config',
                                   'babbling_mode',
                                   'interest_model',
                                   'interest_model_config',
                                   'sensorimotor_model',
                                   'sensorimotor_model_config'))


def make_settings(environment,
                  babbling_mode,
                  interest_model, sensorimotor_model,
                  environment_config='default',
                  interest_model_config='default',
                  sensorimotor_model_config='default'):

    return Settings(environment, environment_config,
                    babbling_mode,
                    interest_model, interest_model_config,
                    sensorimotor_model, sensorimotor_model_config)

from .experiment import Experiment
from .pool import ExperimentPool
