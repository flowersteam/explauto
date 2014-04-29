import itertools

from .experiment import Experiment
# from copy import deepcopy


class ExperimentPool(object):
    def __init__(self, environments, babblings, interest_models, sensorimotor_models):
        configurations = itertools.product(environments, babblings,
                                           interest_models, sensorimotor_models)
        # print list(deepcopy(configurations))
        # print list(deepcopy(configurations))
        # self.pool = list(deepcopy(configurations))

        self.xps = [Experiment.from_settings(env, bab, im, sm, env_conf, im_conf, sm_conf)
                    for (env, env_conf), bab, (im, im_conf), (sm, sm_conf) in configurations]

    def bootstrap(self, n):
        [xp.bootstrap(n) for xp in self.xps]

    def evaluate_at(self, indices, same_evaluation=False):
        for i, xp in enumerate(self.xps):
            xp.evaluate_at(indices)
            if same_evaluation and i > 0:
                xp.evaluation.tester.testcases = self.xps[0].evaluation.tester.testcases
        # self.xps[0].evaluate_at(indices, evaluation=None)
        # if len(self.xps) == 1:
            # return
        # for xp in self.xps[1:]:
            # evaluation.tester.testcases = epcopy(self.xps[0].evaluation) if same_evaluation else None
            # xp.evaluate_at(indices, evaluation=evaluation)

    def run(self, n_iter=-1, bg=False):
        [xp.run(n_iter, bg=bg) for xp in self.xps]

    def wait(self):
        [xp.wait() for xp in self.xps]
