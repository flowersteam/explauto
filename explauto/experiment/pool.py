import itertools

from multiprocessing.pool import Pool, ThreadPool
from numpy import array, hstack
from copy import deepcopy
from numpy import random

from . import Settings
from .experiment import Experiment
from ..environment import environments


def _f(args):
    settings, evaluate_indices, testcases = args

    random.seed()

    xp = Experiment.from_settings(settings)
    xp.evaluate_at(evaluate_indices, testcases)
    xp.run()

    return xp.log


class ExperimentPool(object):
    def __init__(self, settings, evaluate_at,
                 testcases=None, same_testcases=False):
        """ Pool of experiments running in parallel.

            :param list settings: list of settings used to create the pool of experiments.
            :param list evaluate_at: iteration indices where an evaluation should be performed.
            :param numpy.array testcases: specify a pre-defined testcases, by default a testcases will be generated.
            :param bool same_testcases: whether to use the same testcases for all experiments.

            The Pool will create :class:`~explauto.experiment.experiment.Experiment` using the :meth:`~explauto.experiment.experiment.Experiment.from_settings` constructor for each combination of parameters given.

            .. note:: If you set same_testcases to True the first experiment will generate a testcase used by all the others experiment. Otherwise, each experiment will generate its own testcase.
        """
        if same_testcases and testcases is None:
            s = settings[0]
            xp = Experiment.from_settings(s)
            xp.evaluate_at(evaluate_at)
            testcases = xp.evaluation.tester.testcases

        self._config = zip(settings,
                           itertools.repeat(evaluate_at),
                           itertools.repeat(testcases))

    @classmethod
    def from_settings_product(cls, environments, babblings,
                              interest_models, sensorimotor_models,
                              evaluate_at, n_bootstrap=[0], same_testcases=False):
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
                              interest_models, sensorimotor_models, n_bootstrap)

        settings = [Settings(env, env_conf, bab, im, im_conf, sm, sm_conf, n_boot)
                    for ((env, env_conf), bab, (im, im_conf), (sm, sm_conf), n_boot) in l]

        return cls(settings, evaluate_at, same_testcases=same_testcases)

    def run(self, repeat=1, processes=None, use_thread=False):
        """ Runs all experiments using a :py:class:`multiprocessing.Pool`.

            :param int repeat: Number of time each experiment will be repeated.
            :param int processes: Number of processes launched in parallel (Default: uses all the availabled CPUs)
            :param bool use_thread: Use a :py:class:`~multiprocessing.pool.ThreadPool` instead of a :py:class:`~multiprocessing.pool.Pool`. By default, tries to guess depending on the environment which one to use.

         """
        mega_config = [c for c in self._config for _ in range(repeat)]

        env = [environments[s.environment][0] for s in self.settings]
        use_process = array([e.use_process for e in env]).all() and (not use_thread)

        pool = Pool(processes) if use_process else ThreadPool(processes)

        logs = pool.map(_f, mega_config)
        logs = array(logs).reshape(-1, repeat)

        self._add_logs(logs)

        return logs

    @property
    def settings(self):
        """ Returns a copy of the list of all the settings used. """
        return array(self._config)[:, 0].tolist()

    @property
    def logs(self):
        if not hasattr(self, '_logs'):
            raise ValueError('You have to run the pool of experiments first!')

        logs = self._logs.reshape(-1) if self._logs.shape[1] == 1 else self._logs
        return deepcopy(logs)

    def _add_logs(self, logs):
        if not hasattr(self, '_logs'):
            self._logs = logs

        else:
            self._logs = hstack((self._logs, logs))
