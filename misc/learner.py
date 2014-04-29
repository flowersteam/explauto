import configure
import simple_arm as arm
import bayesopt
import bayesoptmodule
import numpy as np
from scipy.linalg import norm
from time import clock

goal = arm.forward([np.pi/3, 5*np.pi/6])
def testfunc(xin):
    print xin
    return norm(goal - arm.forward(xin))

# Class for OO testing.
class BayesOptTest(bayesoptmodule.BayesOptContinuous):
    def evalfunc(self,Xin):
        return testfunc(Xin.T)


# Let's define the parameters
# For different options: see parameters.h and cpp
# If a parameter is not define, it will be automatically set
# to a default value.
params = bayesopt.initialize_params()
params['n_iterations'] = 50
params['n_init_samples'] = 20
#params['surr_name'] = "GAUSSIAN_PROCESS_INV_GAMMA_NORMAL"
params['crit_name'] = "cEI"
params['kernel_name'] = "kMaternISO3"
print "Callback implementation"

n = 2                     # n dimensions
lb = -np.pi*np.ones((n,))
ub = np.pi*np.ones((n,))

bo_test = BayesOptTest()
bo_test.params = params
bo_test.n_dim = n
bo_test.lower_bound = lb
bo_test.upper_bound = ub

start = clock()
mvalue, x_out, error = bo_test.optimize()

print "Result", x_out
print "Seconds", clock() - start
