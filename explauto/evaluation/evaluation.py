from numpy import linalg


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
        mode = self.ag.sensorimotor_model.mode
        self.ag.sensorimotor_model.mode = 'exploit'
        if self.mode == 'inverse':
            errors = []
            for s_g in self.testcases:
                m = self.ag.infer(self.ag.conf.s_dims, self.ag.conf.m_dims, s_g).flatten()
                s = self.env.update(m, log=False)
                errors.append(linalg.norm(s_g - s))
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
                s = self.env.update(m, log=False)
                errors.append(linalg.norm(s_p - s))
        else:
            raise ValueError('mode should be "inverse" or "forward"',
                              '"general" predictions coming soon)')

        self.ag.sensorimotor_model.mode = mode
        return errors

    def plot_testcases(self, ax, dims, **kwargs_plot):
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        # test_array = array([hstack((m, s)) for m, s in self.tester.testcases])
        # test_array = test_array[:, dims]
        # ax.plot(*(test_array.T), **plot_specs)
        ax.plot(*(self.testcases.T), **plot_specs)
