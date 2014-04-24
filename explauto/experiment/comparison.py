import itertools

from .experiment import Experiment


class ExperimentPool(object):
    def __init__(self, environments, babblings, interest_models, sensorimotor_models):
        self.configurations = itertools.product(environments, babblings,
                                                interest_models, sensorimotor_models)

        self.xps = [Experiment.from_settings(env, bab, im, sm, env_conf, im_conf, sm_conf)
                    for (env, env_conf), bab, (im, im_conf), (sm, sm_conf) in self.configurations]

    def bootstrap(self, n):
        [xp.bootstrap(n) for xp in self.xps]

    def evaluate_at(self, indices):
        [xp.evaluate_at(indices) for xp in self.xps]

    def run(self, nb_iter=-1, bg=False):
        [xp.run(nb_iter, bg=bg) for xp in self.xps]

    def wait(self):
        [xp.wait() for xp in self.xps]
