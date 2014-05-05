import itertools

from explauto.experiment import Experiment
from multiprocessing import Pool


def _f(args):
    (env, env_conf), bab, (im, im_conf), (sm, sm_conf), eval_ind, testcases = args

    xp = Experiment.from_settings(env, bab, im, sm, env_conf, im_conf, sm_conf)
    xp.evaluate_at(eval_ind, testcases)
    xp.bootstrap(5)
    xp.run()

    return xp.logs


class ExperimentPool(object):
    def __init__(self, environments, babblings, interest_models, sensorimotor_models,
                 evaluate_at, same_testcases=False):
        """ Pool of experiments running in parallel.

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

            The Pool will create :class:`~explauto.experiment.experiment.Experiment` using the :meth:`~explauto.experiment.experiment.Experiment.from_settings` constructor for each combination of parameters given.

            .. note:: If you set same_testcases to True the first experiment will generate a testcase used by all the others experiment. Otherwise, each experiment will generate its own testcase.

        """
        if same_testcases:
            # We create a dummy environment just to generate the testcase
            env, env_conf = environments[0]
            bab = babblings[0]
            im, im_conf = interest_models[0]
            sm, sm_conf = sensorimotor_models[0]

            xp = Experiment.from_settings(env, bab, im, sm,
                                          env_conf, im_conf, sm_conf)
            xp.evaluate_at(evaluate_at)
            testcases = xp.evaluation.tester.testcases

        else:
            testcases = None

        self._config = list(itertools.product(environments, babblings,
                                              interest_models, sensorimotor_models,
                                              [evaluate_at], [testcases]))
        self.logs = []

    def run(self, processes=None):
        """ Runs all experiments using a :py:class:`multiprocessing.Pool`.

            :param int processes: Number of processes launched in parallel (Default: uses all the availabled CPUs)
         """
        return Pool(processes).map(_f, self.configurations)

    @property
    def configurations(self):
        """ Returns a copy of the list of all the configurations used. """
        return list(self._config)
