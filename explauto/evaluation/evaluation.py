from ..third_party.models.models.testbed import Testbed
from ..adaptors import Learner, Robot


class Evaluation(object):
    def __init__(self, ag, env, n_samples=100, mode='inverse'):
        learner = Learner(ag)
        robot = Robot(env)
        self.tester = Testbed(robot, learner)
        self.mode = mode
        self.errors = []
        if self.mode == 'forward':
            self.tester.uniform_motor(n_samples)
        elif self.mode == 'inverse':
            self.tester.uniform_sensor(n_samples)
        else:
            print 'mode should be "forward" or "inverse" (general prediction coming soon)'
            raise ValueError

    def evaluate(self):
        if self.mode == 'forward':
            self.errors = self.tester.run_forward()
        elif self.mode == 'inverse':
            self.errors = self.tester.run_inverse()
        return self.errors
