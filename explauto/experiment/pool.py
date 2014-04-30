import itertools
import threading

from explauto.experiment import Experiment
from multiprocessing import Pool


def f(args):
    (env, env_conf), bab, (im, im_conf), (sm, sm_conf), eval_ind, testcases = args

    xp = Experiment.from_settings(env, bab, im, sm, env_conf, im_conf, sm_conf)
    xp.evaluate_at(eval_ind, testcases)
    xp.bootstrap(5)
    xp.run()

    return xp.logs


class ExperimentPool(object):
    def __init__(self, environments, babblings, interest_models, sensorimotor_models,
                 evaluate_at, same_testcases=False):

            if same_testcases:
                env, env_conf = environments[0]
                bab = babblings[0]
                im, im_conf = interest_models[0]
                sm, sm_conf = sensorimotor_models[0]

                xp = Experiment.from_settings(env, bab, im, sm, env_conf, im_conf, sm_conf)
                xp.evaluate_at(evaluate_at)
                testcases = xp.evaluation.tester.testcases

            else:
                testcases = None

            self.configurations = list(itertools.product(environments, babblings,
                                                         interest_models, sensorimotor_models,
                                                         [evaluate_at], [testcases]))
            self.logs = []

    def run(self):
        return Pool().map(f, self.configurations)
