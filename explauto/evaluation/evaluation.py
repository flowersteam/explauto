from numpy import array, hstack

from ..third_party.models.models.testbed import Testbed
from ..third_party.models_adaptors import Learner, Robot


class Evaluation(object):
    def __init__(self, ag, env, testcases=None, n_samples=100, mode='inverse'):
        self.ag = ag
        learner = Learner(ag)
        robot = Robot(env, log=False)
        self.tester = Testbed(robot, learner, testcases)
        self.mode = mode
        self.errors = []

        if testcases is None:
            if mode not in ('forward', 'inverse'):
                raise ValueError('mode should be "forward" or "inverse"'
                                 '(general prediction coming soon)')

            tests = getattr(self.tester,
                            'uniform_motor' if mode == 'forward' else 'uniform_sensor')
            tests(n_samples)

    def evaluate(self):
        mode = self.ag.sensorimotor_model.mode
        self.ag.sensorimotor_model.mode = 'exploit'
        if self.mode == 'forward':
            self.errors = self.tester.run_forward()
        elif self.mode == 'inverse':
            self.errors = self.tester.run_inverse()
        self.ag.sensorimotor_model.mode = mode
        return self.errors

    def plot_testcases(self, ax, dims, **kwargs_plot):
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        test_array = array([hstack((m, s)) for m, s in self.tester.testcases])
        test_array = test_array[:, dims]
        ax.plot(*(test_array.T), **plot_specs)
