from ..third_party.models.models.testbed import Testbed
from ..third_party.models_adaptors import Learner, Robot


class Evaluation(object):
    def __init__(self, ag, env, n_samples=100, mode='inverse'):
        self.ag = ag
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
            raise ValueError('mode should be "forward" or "inverse"'
                             '(general prediction coming soon)')

    def evaluate(self):
        mode = self.ag.sensorimotor_model.mode
        self.ag.sensorimotor_model.mode = 'exploit'
        if self.mode == 'forward':
            self.errors = self.tester.run_forward()
        elif self.mode == 'inverse':
            self.errors = self.tester.run_inverse()
        self.ag.sensorimotor_model.mode = mode
        return self.errors
