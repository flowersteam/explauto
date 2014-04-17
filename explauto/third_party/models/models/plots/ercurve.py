import string

import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.get_cmap('gist_rainbow')

from toolbox import gfx
from .. import testbed

class ErrorCurve(object):

    def __init__(self, robot, learners, side = 'forward', trials = 1000, test_period = 1,  test_pace = 1.5, tests = 100, uniformity = 'sensor',
                 subplot = None, caption = True, extent = None):
        """
        @param trials  how many trials
        @param factor  how test are distributed along time, following a i**factor, i = 2,3,4... serie
        @param tests   how many tests
        """
        self.side = side
        self.robot = robot
        self.learners = learners
        self.testbeds = [testbed.Testbed.from_learner(robot, learnr) for learnr in learners]
        self.trials  = trials
        assert test_pace   >= 1.0
        assert test_period >= 1.0
        self.test_period = test_period
        self.test_pace = test_pace
        self.tests = tests
        assert uniformity == 'sensor' or uniformity == 'motor'
        self.uniformity = uniformity
        self.subplot = subplot
        self.disp_caption = caption

    def _caption(self):
        self.title = 'Benchmarking %s Models' % (string.upper(self.side[0]) + self.side[1:])
        s  = ''
        s += 'Average error of different %s models on the robot %s in function of the number of training examples.\n' % (self.side, self.robot.name)
        s += '%i tests uniformly distributed in the %s space.\n' % (self.tests, self.uniformity)
        s += 'Robot config <%s>\n' % (self.robot)
        for i, l in enumerate(self.learners):
            if self.side == 'forward':
                fwd = l.imodel.fmodel
                s += '%i. %s with <%s>\n' % (i+1, fwd.name, fwd.config())
            else:
                inv = l.imodel
                fwd = l.imodel.fmodel
                s += '%i. %s with <%s> + %s with <%s>\n' % (i+1, inv.name, inv.config(), fwd.name, fwd.config())
        return s

    def _setup(self, common_dataset = True):
        # Sharing dataset
        tb0 = self.testbeds[0]
        if common_dataset:
            for tb in self.testbeds:
                tb.fmodel.dataset = tb0.fmodel.dataset

        # Sharing testcases
        if self.uniformity == 'sensor':
            tb0.uniform_sensor(self.tests)
        else:
            tb0.uniform_motor(self.tests)
        for tb in self.testbeds:
            tb.testcases = tb0.testcases

        # Creating teststeps, error vectors
        self._teststeps = []
        i, k = 0, 0
        while k < self.trials:

            self._teststeps.append(k)
            k = int(self.test_period*(i+1)**self.test_pace)
            i += 1
        self._teststeps.append(self.trials-1)
        print self._teststeps
        #print self._teststeps

        #self._teststeps    = [int(i**self.test_pace) for i in range(1, int(self.trials**(1.0/self.test_pace)))]
        self._errors_avg   = [[] for tb in self.testbeds]
        self._errors_lwstd = [[] for tb in self.testbeds]
        self._errors_upstd = [[] for tb in self.testbeds]

    def _run_test(self, tb, j, k):
        if self.side == 'forward':
            errors = tb.run_forward()
        else:
            errors = tb.run_inverse()
        done = (j+1) + k*len(self.testbeds)
        gfx.print_progress(done, len(self._teststeps)*len(self.testbeds),
                           prefix = "Running tests ", quiet = False,
                           eta = 1, freq = 0.5)
        return errors

    def _process_errors(self, tb, errors, j):
        avg, lwstd, upstd = tb.avg_std_asym(errors)
        self._errors_avg[j].append(avg)
        self._errors_lwstd[j].append(avg - lwstd)
        self._errors_upstd[j].append(avg + upstd)

    def _test(self, k):
        for j, tb in enumerate(self.testbeds):
            errors = self._run_test(tb, j, k)
            self._process_errors(tb, errors, j)

    def _run(self):
        # Training and Testing
        tb0 = self.testbeds[0]
        k = 0
        for i in range(self.trials):
            tb0.train_motor(1)
            if i in self._teststeps:
                self._test(k)
                k += 1

        print('')

    def _subplot(self):
        # Plot
        if self.subplot is not None:
            plt.subplot(self.subplot)
        else:
            plt.subplot(111)
        plt.xlabel('timesteps')
        #plt.title(self.caption(), fontsize=10)
        plt.ylabel('average error')
        cutoff = 0
        for i, tb in enumerate(self.testbeds):
            color = cmap(float(i)/len(self.testbeds))
            print len(self._teststeps[cutoff:]), len(self._errors_avg[i][cutoff:])
            print len(self._teststeps), len(self._errors_avg[i])
            plt.plot(self._teststeps[cutoff:], self._errors_avg[i][cutoff:], color = color)
            plt.fill_between(self._teststeps, self._errors_lwstd[i], self._errors_upstd[i], facecolor=color, alpha=0.2)

        if self.disp_caption:
            plt.subplots_adjust(left=0.1, right=0.9, top= 0.75, bottom=0.1)
            plt.figtext(0.1, 0.8, self._caption(), fontsize=8)
            # Legend
            if self.side == 'forward':
                leg = plt.legend(['%i. ' % (i+1, ) + tb.fmodel.name for i, tb in enumerate(self.testbeds)],
                                 'upper right', shadow=False)
            else:
                leg = plt.legend(['%i. ' % (i+1, ) + str(tb.imodel.name + ' + ' + tb.imodel.fmodel.name) for i, tb in enumerate(self.testbeds)],
                                 'upper right', shadow=False)
            for t in leg.get_texts():
                t.set_fontsize(8)


    def _plot(self):
        self._subplot()
        fig = plt.figure(1)
        fig.set_facecolor('white')
        fig.suptitle(self.title)

    def plot(self):
        self._setup()
        self._run()
        self._plot()

    def show(self):
        plt.show()
