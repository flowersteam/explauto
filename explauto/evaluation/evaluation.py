from numpy import array, hstack
from numpy import linalg

from ..third_party.models.models.testbed import Testbed
from ..third_party.models_adaptors import Learner, Robot


class Evaluation(object):
    def __init__(self, ag, env, testcases, n_samples=100, mode='inverse'):
        self.ag = ag
        self.env = env
        # learner = Learner(ag)
        # robot = Robot(env, log=False)
        # self.tester = Testbed(robot, learner, testcases)
        self.mode = mode
        # self.errors = []

        if mode not in ('inverse'):
            raise ValueError('mode should be "inverse"'
                             '("forward" and "general" predictions coming soon)')
        self.testcases = testcases

            # tests = getattr(self.tester,
                            # 'uniform_motor' if mode == 'forward' else 'uniform_sensor')
            # tests(n_samples)

    def evaluate(self):
        mode = self.ag.sensorimotor_model.mode
        self.ag.sensorimotor_model.mode = 'exploit'
        if self.mode == 'inverse':
            errors = []
            for s_g in self.testcases:
                m = self.ag.infer(self.ag.conf.s_dims, self.ag.conf.m_dims, s_g)
                self.env.update(m, log=False)
                s = self.env.state[self.env.conf.s_dims]
                errors.append(linalg.norm(s_g - s))
        else:
            raise ValueError('mode should be "inverse"'
                             '("forward" and "general" predictions coming soon)')

        self.ag.sensorimotor_model.mode = mode
        return errors

    def plot_testcases(self, ax, dims, **kwargs_plot):
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        # test_array = array([hstack((m, s)) for m, s in self.tester.testcases])
        # test_array = test_array[:, dims]
        # ax.plot(*(test_array.T), **plot_specs)
        ax.plot(*(self.testcases.T), **plot_specs)
