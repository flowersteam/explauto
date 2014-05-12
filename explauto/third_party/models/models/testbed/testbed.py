"""Systematic testing of an experiment"""

import numpy as np

from ....toolbox import toolbox
from . import testcase

from explauto import ExplautoEnvironmentUpdateError


class Testbed(object):

    @classmethod
    def from_learner(cls, robot, learner):
        """"""
        return cls(robot, imodel=learner.imodel)

    def __init__(self, robot, learner, testcases=[]):
        """
        @param fmodel  forward model. If not provided, will look for the
                       'fmodel' attribute in the inverse model.
        @param imodel  inverse model. optional if forward model is provided.
                       (and run_inverse is not called !)
        """
        self.learner = learner
        # assert fmodel is not None or imodel is not None
        self.robot = robot
        # self.fmodel = fmodel
        # if fmodel is None:
        #    self.fmodel = imodel.fmodel
        #   self.imodel = imodel
        self.testcases = testcases

    # Training

    def train_motor(self, n):
        """Train the models on n trials uniformaly distributed in the motor space"""
        tests = testcase.uniform_motor_testcases(self.robot, n)
        if self.imodel is not None:
            for x, y in tests:
                self.imodel.add_xy(x, y)
        if self.fmodel is not self.imodel.fmodel:
            for x, y in tests:
                self.fmodel.add_xy(x, y)

    def train_sensor(self, n):
        """Train the models on n trials uniformaly distributed in the sensory space"""
        tests = testcase.uniform_sensor_testcases(self.robot, n)
        if self.imodel is not None:
            for x, y in tests:
                self.imodel.add_xy(x, y)
        if self.fmodel is not self.imodel.fmodel:
            for x, y in tests:
                self.fmodel.add_xy(x, y)

    # Test

    def reset_testcases(self):
        self.testcases = []

    def uniform_motor(self, n):
        """Generate n test uniformly distributed in the motor space"""
        self.testcases = testcase.uniform_motor_testcases(self.robot, n)
        return self.testcases

    def uniform_sensor(self, n):
        """Generate n test uniformly (approximately) distributed in the sensory space"""
        self.testcases = testcase.uniform_sensor_testcases(self.robot, n)
        return self.testcases

    def run_forward(self):
        """Run the tests
        @return list of error of each tests
        """
        errors = []
        for order, effect in self.testcases:
            #predicted_effect = self.fmodel.predict_y(order)
            predicted_effect = self.learner.predict_effect(order)
            errors.append(toolbox.dist(predicted_effect, effect))
        return errors

    def run_inverse(self):
        """Run the tests. Note that the robot will execute an order for every test.
        @return list of error of each tests
        """
        # assert self.imodel is not None
        errors = []
        for order, effect in self.testcases:
            predicted_order = self.learner.infer_order(effect)
            try:
                obtained_effect = self.robot.execute_order(predicted_order)
                errors.append(toolbox.dist(obtained_effect, effect))
            except ExplautoEnvironmentUpdateError:
                errors.append(np.inf)
        return errors

    def avg_std(self, errors):
        """Return average and standard deviation of the error provided."""
        return np.average(errors), np.std(errors)

    def avg_std_asym(self, errors):
        """Return mean and the asymetric std."""
        mean = np.average(errors)
        upstd = np.std([e for e in errors if e >= mean])
        lowstd = np.std([e for e in errors if e <= mean])
        return mean, lowstd, upstd


# # Saving OLD CODE TO ADAPT
#
# def save_unitests(featslist, expname, n = 10000, res = 10):
#     """Create uniform tests from drawing n random observations"""
#     obsstream = packs.ObsStream(packs.load('obs10000', 'packs/%s' % expname))
#     assert obsstream.probablesize > n*len(featslist)
#
#     for feats in featslist:
#         tests  = pre.unitt.make_unitests(n, feats, obsstream.next, res = res)
#         print ("%i obs at %i resolution for feats %s -> %i arm goals, %i ball goals"
#                % (n, res, feats, len(tests)))
#         pack = {}
#         pack['metadata'] = {'desc' : 'Uniformly distributed tests',
#                             'feats': feats,
#                             'res'  : RES,
#                             'pool' : N,
#                             'size' : len(tests)}
#         pack['data']     = tests
#
#         filepath = paths.fullpath('packs/%s/unitt%s[%s.%s].ppy'
#                                    % (expname, feats, N, RES),
#                                   location = paths.local)
#         with open(filepath, 'w') as f:
#             cPickle.dump(pack, f, -1)
#
