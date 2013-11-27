"""Benchmarking different forward models in function of timesteps."""

import sys
import random
random.seed(0)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
cmap = cm.get_cmap('gist_rainbow')

import robots
import models.learner as learner
import models.testbed as testbed

trials = 3000 # How many trials
factor = 1.5  # How test are distributed along time, following a i**factor, i = 2,3,4... serie
tests  = 250  # How many tests

def fwdcurves(robot, n):
    # Instanciating testbed
    fwdlearners = [learner.Learner.from_robot(robot, fwd = fwd, inv = 'NN', k = 20) for fwd in learner.fwdclass]
    testbeds    = [testbed.Testbed.from_learner(robot, learnr) for learnr in fwdlearners]

    # Sharing dataset
    tb0 = testbeds[0]
    for tb in testbeds:
        tb.fmodel.dataset = tb0.fmodel.dataset

    # Creating Teststep
    teststeps = [int(i**factor) for i in range(1, int(trials**(1.0/factor)))]

    # Sharing testcases
    tb0.uniform_motor(tests)
    for tb in testbeds:
        tb.testcases = tb0.testcases

    # Training and Testing
    errors_avg  = [[] for tb in testbeds]
    lower_bound = [[] for tb in testbeds]
    upper_bound = [[] for tb in testbeds]
    for i in range(trials):
        tb0.train_motor(1)
        if i in teststeps:
            print "Testing at step %i/%i     \r" % (i, trials),
            sys.stdout.flush()
            for j, tb in enumerate(testbeds):
                errors = tb.run_forward()
                avg, std = tb.avg_std(errors)
                errors_avg[j].append(avg)
                lower_bound[j].append(avg-std)
                upper_bound[j].append(avg+std)

    # Plot
    lines = []
    plt.subplot(n)
    plt.xlabel('timesteps')
    plt.title(robot.__class__.__name__)
    plt.ylabel('average error')
    for i, tb in enumerate(testbeds):
        lines.append(plt.plot(teststeps, errors_avg[i]))
        plt.fill_between(teststeps, lower_bound[i], upper_bound[i], facecolor=color, alpha=0.2)



    return lines, tuple(tb.fmodel.__class__.__name__ for tb in testbeds)

# Robot
arm3 = robots.KinematicArm2D(dim = 3)
arm6 = robots.KinematicArm2D(dim = 6)
vm   = robots.VowelModel()
ergo = robots.Ergorobot()

gs = gridspec.GridSpec(2, 2)
gs.update(wspace = 0.4, hspace = 0.4)
handles, labels = fwdcurves(arm3, gs[0, 0])
handles, labels = fwdcurves(arm6, gs[0, 1])
handles, labels = fwdcurves(vm,   gs[1, 0])
handles, labels = fwdcurves(ergo, gs[1, 1])

fig = plt.figure(1)
fig.set_facecolor('white')
fig.suptitle('Benchmarks of Forward Model for Differents Robots')
leg = fig.legend(handles, labels, 'lower right',)
for t in leg.get_texts():
    t.set_fontsize('small')


plt.show()