import testenv

import random

try:
	import robots
	robotsimported = True
except Exception:
	print('%s %s' % ('\033[1;33mSKIP\033[0m', 'robots package import failed. tests not run.'))
	robotsimported = False

import models.learner as learner

def test_curve_coverage():
	"""Instanciate all forward models to being tested."""
	# Robots
	import robots
	arm = robots.KinematicArm2D(dimM = 3)

	# Learners
	fwdlearners = [learner.Learner.from_robot(vm, fwd = fwd, inv = 'NN') for fwd in learner.fwdclass.keys()]
	curve = ErrorCurve(arm, fwdlearners, side = 'forward', trials = 30, tests = 2, uniformity = 'sensor')
	curve.plot()

	return True

def test_learner_params():
	"""Check if parameter given in learner intialization trickle down to models"""
	check = True

	for i in range(10):
		sigma = random.uniform(0.1, 1000.0)
		l = learner.Learner((-2, -1), (0, 1), ((0, 1), (0, 1)), fwd = 'LWLR', sigma = sigma)
		check = check and l.imodel.fmodel.sigma == sigma

	for i in range(10):
		k = random.randint(1, 100)
		l = learner.Learner((-2, -1), (0, 1), ((0, 1), (0, 1)), fwd = 'WNN', k = k)
		check = check and l.imodel.fmodel.k == k

	return check

tests = [test_learner_coverage,
		 test_learner_params]

if __name__ == "__main__" and robotsimported:
	print("\033[1m%s\033[0m" % (__file__,))
	for t in tests:
		print('%s %s' % ('\033[1;32mPASS\033[0m' if t() else
						 '\033[1;31mFAIL\033[0m', t.__doc__))