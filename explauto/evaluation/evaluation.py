from numpy import linalg, array, zeros, sum, min, max


class Evaluation(object):
    def __init__(self, ag, env, testcases, mode='inverse'):
        self.ag = ag
        self.env = env
        self.mode = mode

        if mode not in ('inverse', 'forward'):
            raise ValueError('mode should be "inverse" or "forward"',
                             '"general" predictions coming soon)')
        self.testcases = testcases

    def evaluate(self, n_tests_forward=None, testcases_forward=None):
        print "Evaluation mode : ", self.mode
        self.ag.eval_mode()
        s_reached = []
        if self.mode == 'inverse':
            errors = []
            for s_g in self.testcases:
                m = self.ag.infer(self.ag.config.eval_dims, self.ag.conf.m_dims, s_g, pref = 'eval_').flatten()
                m_env = self.ag.motor_primitive(m)
                s_env = self.env.update(m_env, log=False)
                s_ = self.ag.sensory_primitive(s_env)
                s = self.ag.get_eval_dims(s_)
                e = linalg.norm(s_g - s)
                s_reached.append(s)
                errors.append(e)
                print 'Evaluation', len(errors), ': s_goal = ', s_g, 's_reached = ', s_, 'L2 error = ', e, '\n'
        elif self.mode == 'forward':
            print 'forward prediction tests still in beta version, use with caution'
            if n_tests_forward is not None:
                print "Generating ", n_tests_forward, " uniform random motor tests ..."
                testcases = self.env.random_motors(n=n_tests_forward)
            elif testcases_forward is not None:
                testcases = testcases_forward
            else:
                raise ValueError('For forward prediction evaluation',
                                  ', call either using n_tests_forward',
                                  '(# of uniform random motor tests) or',
                                  'testcases_forward (motor testcases). Not both.')
            errors = []
            for m in testcases:
                s_p = self.ag.infer(self.ag.conf.m_dims, self.ag.conf.s_dims, m).flatten()
                m_env = self.ag.motor_primitive(m)
                s_env = self.env.update(m_env, log=False)
                s = self.ag.sensory_primitive(s_env)
                errors.append(linalg.norm(s_p - s))
                s_reached.append(s)
        else:
            raise ValueError('mode should be "inverse" or "forward"',
                              '"general" predictions coming soon)')

        self.ag.learning_mode()
        print s_reached
        return errors,s_reached
    

    def evaluate_explo(self, log):
        
        
        sx = array([a[0] for a in log.logs['perception_mod'+'{}'.format(len(log.config.mids))]])
        sy = array([a[1] for a in log.logs['perception_mod'+'{}'.format(len(log.config.mids))]])
        
        n_eval_dims = len(log.config.eval_dims)
        eval_range = array([[min(sx),min(sy)],
                           [max(sx),max(sy)]])
        
        eps = 0.01#log.config.eval_explo_eps
        grid_sizes = (eval_range[1,:] - eval_range[0,:]) / eps + 1
        grid_sizes = array(grid_sizes, dtype = int)
        grid = zeros(grid_sizes)
        for i in range(len(sx)):
            idx = int((sx[i] - eval_range[0,0]) / eps)
            idy = int((sy[i] - eval_range[0,1]) / eps)
            grid[idx, idy] = grid[idx, idy] + 1
        grid[grid > 1] = 1
        explo = sum(grid)
        
        return explo

    def plot_testcases(self, ax, dims, **kwargs_plot):
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        # test_array = array([hstack((m, s)) for m, s in self.tester.testcases])
        # test_array = test_array[:, dims]
        # ax.plot(*(test_array.T), **plot_specs)
        ax.plot(*(self.testcases.T), **plot_specs)
