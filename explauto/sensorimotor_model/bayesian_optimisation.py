from .sensorimotor_model import SensorimotorModel
from explauto.models.dataset import BufferedDataset as Dataset
import GPy
import GPyOpt
import numpy as np
from GPyOpt.util.general import reshape

class F_to_minimize(object):
    ''' Class used during the optimisation '''
    def __init__(self, s_goal, input_dim, environment):
        ''' :param scalar tuple s_goal: sensoriel goal of the optimisation
            :param scalar input_dim: size of the input, dimension of the motor space
            :param Environment environment: environment on which is use the optimisation
        '''
        self.s_goal = s_goal
        self.input_dim = input_dim
        self.pointsList = []
        self.environment = environment
    def dist(self, s):
        ''' Distance to the goal, use the euclidian distance for now '''
        #~ return [np.exp(np.linalg.norm(s-self.s_goal)**2)]
        return [np.linalg.norm(s-self.s_goal)]
    def dist_array(self, S):
        return np.array(map(self.dist,S))
    def f(self, m):
        ''' Function to minimize '''
        M = reshape(m,self.input_dim)
        n = M.shape[0]
        if M.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            S = self.environment.update(M)
            m = M[0]
            s = S[0]
            # Store the points (m,s) explored during the optimisation
            self.pointsList.append((m,s))
            return self.dist_array(S)


class BayesianOptimisation(SensorimotorModel):
    ''' Sensorimotor model using Bayesian optimisation to infer inverse prediction'''
    def __init__(self, conf, acquisition, exploration_weight, initial_points, environment, optimisation_iterations, exact_feval):
        ''' :param string acquisition: choose the model of acquisition function between "MPI","EI" and "LCB"
            :param scalar exploration_weight: module the exploration in the acquistion (base 2 for LCB and 0.01 for others)
            :param scalar initial_points: the number of initial points to give for the bayesian optimisation
            :pram Environment environment: environment on which is used the optimisation
            :param scalar optimisation_iterations: number of iterations of the optimisation
            :param boolean exact_feval: must be False if the environment is noisy
        '''
        for attr in ['m_dims', 's_dims']:
            setattr(self, attr, getattr(conf, attr))
        self.dim_x = len(self.m_dims)
        self.dim_y = len(self.s_dims)
        self.acquisition = acquisition
        self.exploration_weight = exploration_weight
        self.initial_points = initial_points
        self.dataset  = Dataset(len(self.m_dims), len(self.s_dims))
        self.conf = conf
        self.mode = 'explore'
        self.environment = environment
        self.optimisation_iterations = optimisation_iterations
        self.exact_feval = exact_feval

    def infer(self, in_dims, out_dims, x):
        if in_dims == self.m_dims and out_dims == self.s_dims:    # forward
            ''' For now only return the nearest neighbor of the motor action '''
            assert len(x) == self.dim_x, "Wrong dimension for x. Expected %i, got %i" % (self.dim_x, len(x))
            # Find the nearest neighbor of the motor action x
            _, index = self.dataset.nn_x(x, k=1)
            return self.dataset.get_y(index[0])

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if self.mode == 'exploit':
                self.acquisition = 0
            assert len(x) == self.dim_y, "Wrong dimension for x. Expected %i, got %i" % (self.dim_y, len(x))

            # Find the motor action that lead to the nearest neighbor of the sensitive goal
            _, index = self.dataset.nn_y(x, k=1)
            x0 = self.dataset.get_x(index[0])
            # Find the k nearest neighbors of this motor action
            _, index = self.dataset.nn_x(x0, k=self.initial_points)
            X = []
            Y = []
            for i in range(len(index)):
                X.append(self.dataset.get_x(index[i]))
                Y.append(self.dataset.get_y(index[i]))

            # Initialisation of the Bayesian optimisation
            func = F_to_minimize(np.array(x), self.dim_x, self.environment)
            X_init = np.array(X)
            Y_init = func.dist_array(Y)
            bounds = []
            for i in range(self.dim_x):
                bounds.append({'name': 'var_'+  str(i), 'type': 'continuous', 'domain': [self.conf.m_mins[i],self.conf.m_maxs[i]]})
            space                 = GPyOpt.Design_space(bounds)
            objective             = GPyOpt.core.task.SingleObjective(func.f)
            model                 = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False, exact_feval = self.exact_feval)
            acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
            if self.acquisition == 'EI':
                acquisition       = GPyOpt.acquisitions.AcquisitionEI(model, space, acquisition_optimizer, jitter = self.exploration_weight)
            elif self.acquisition == 'MPI':
                acquisition       = GPyOpt.acquisitions.AcquisitionMPI(model, space, acquisition_optimizer, jitter = self.exploration_weight)
            elif self.acquisition == 'LCB':
                acquisition       = GPyOpt.acquisitions.AcquisitionLCB(model, space, acquisition_optimizer, exploration_weight = self.exploration_weight)
            else:
                raise NotImplementedError
            evaluator             = GPyOpt.core.evaluators.Sequential(acquisition)
            bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, X_init = X_init, Y_init = Y_init)

            # Run the optimisation, the eps = -np.inf is set to force the optimisation to do the required number of iterations
            bo.run_optimization(max_iter = self.optimisation_iterations, eps = -np.inf)

            # Update the model woth the list of points explored during the optimisation
            self.list_s = []
            for (m,s) in func.pointsList:
                self.update(m,s)
            return bo.x_opt

        else:
            raise NotImplementedError


    def update(self, m, s):
        self.dataset.add_xy(m, s)

    def forward_prediction(self, m):
        """ Compute the expected sensory effect of the motor command m. It is a shortcut for self.infer(self.conf.m_dims, self.conf.s_dims, m)
        """
        return self.infer(self.conf.m_dims, self.conf.s_dims, m)


    def inverse_prediction(self, s_g):
        """ Compute a motor command to reach the sensory goal s_g. It is a shortcut for self.infer(self.conf.s_dims, self.conf.m_dims, s_g)
        """
        return self.infer(self.conf.s_dims, self.conf.m_dims, s_g)


