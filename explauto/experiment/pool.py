import itertools

from multiprocessing import Pool
from numpy.random import seed
from copy import deepcopy
from numpy import array

from . import Settings
from .experiment import Experiment


def _f(args):
    settings, evaluate_indices, testcases = args

    seed()

    xp = Experiment.from_settings(settings)
    xp.evaluate_at(evaluate_indices, testcases)
    xp.bootstrap(5)
    xp.run()

    return xp.logs


class ExperimentPool(object):
    def __init__(self, settings, evaluate_at, same_testcases=False):
        """ Pool of experiments running in parallel.

            The Pool will create :class:`~explauto.experiment.experiment.Experiment` using the :meth:`~explauto.experiment.experiment.Experiment.from_settings` constructor for each combination of parameters given.

            .. note:: If you set same_testcases to True the first experiment will generate a testcase used by all the others experiment. Otherwise, each experiment will generate its own testcase.
        """
        if same_testcases:
            s = settings[0]
            xp = Experiment.from_settings(s)
            xp.evaluate_at(evaluate_at)
            testcases = xp.evaluation.tester.testcases

        else:
            testcases = None

        self._config = zip(settings,
                           itertools.repeat(evaluate_at),
                           itertools.repeat(testcases))

    @classmethod
    def from_settings_product(cls, environments, babblings,
                              interest_models, sensorimotor_models,
                              evaluate_at, same_testcases=False):
        """ Creates a ExperimentPool with the product of all the given settings.

            :param environments: e.g. [('simple_arm', 'default'), ('simple_arm', 'high_dimensional')]
            :type environments: list of (environment name, config name)
            :param babblings: e.g. ['motor', 'goal']
            :type bablings: list of babbling modes
            :param interest_models: e.g. [('random', 'default')]
            :type interest_models: list of (interest model name, config name)
            :param sensorimotor_models: e.g. [('non_parametric', 'default')]
            :type sensorimotor_models: list of (sensorimotor model name, config name)
            :param evaluate_at: indices defining when to evaluate
            :type evaluate_at: list of int
            :param bool same_testcases: whether to use the same testcases for all experiments

        """
        l = itertools.product(environments, babblings,
                              interest_models, sensorimotor_models)

        settings = [Settings(env, env_conf, bab, im, im_conf, sm, sm_conf)
                    for ((env, env_conf), bab, (im, im_conf), (sm, sm_conf)) in l]

        return cls(settings, evaluate_at, same_testcases)

    def run(self, repeat=1, processes=None):
        """ Runs all experiments using a :py:class:`multiprocessing.Pool`.

            :param int processes: Number of processes launched in parallel (Default: uses all the availabled CPUs)
         """
        mega_config = [c for c in self._config for _ in range(repeat)]

        logs = Pool(processes).map(_f, mega_config)
        # logs = map(_f, mega_config)

        if repeat > 1:
            logs = array(logs).reshape(-1, repeat).tolist()

        self._logs = logs

        return self.logs

    @property
    def settings(self):
        """ Returns a copy of the list of all the settings used. """
        return array(self._config)[:, 0].tolist()

    @property
    def logs(self):
        if not hasattr(self, '_logs'):
            raise ValueError('You have to run the pool of experiments first!')

        return deepcopy(self._logs)
