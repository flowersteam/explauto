import numpy as np
from numpy import linalg


class Evaluation(object):
    def __init__(self, ag, env, testcases, mode='inverse'):
        self.ag = ag
        self.env = env
        self.mode = mode

        if mode not in ('inverse', 'forward', 'mdmsds', 'mcs'):
            raise ValueError('mode should be "inverse" or "forward"',
                             '"general" predictions coming soon)')
        self.testcases = testcases

    def evaluate(self, n_tests_forward=None, testcases_forward=None):
        sm_mode = self.ag.sensorimotor_model.mode
        self.ag.sensorimotor_model.mode = 'exploit'
        if self.mode == 'inverse':
            errors = []
            for s_g in self.testcases:
                m = self.ag.infer(self.ag.conf.s_dims, self.ag.conf.m_dims, s_g).flatten()
                s = self.env.update(m, log=False, reset=True)
                errors.append(linalg.norm(s_g - s))
        elif self.mode == 'mdmsds':
            self.env.reset()
            errors = []
            self.env.reset()
            for s_g in self.testcases:
                self.env.reset()
                
                m = self.env.current_motor_position
                s = self.env.current_sensori_position
                in_dims = range(self.ag.conf.m_ndims/2) + range(self.ag.conf.m_ndims, self.ag.conf.m_ndims + self.ag.conf.s_ndims)
                out_dims = range(self.ag.conf.m_ndims/2, self.ag.conf.m_ndims)
                dm = self.ag.infer(in_dims, 
                                out_dims, 
                                np.array(list(m) + list(np.hstack((s, s_g[len(s_g)/2:])))))
                mdm = np.hstack((m, dm))
                sds = self.env.update(mdm, reset=False)
                errors.append(linalg.norm(s_g[len(s_g)/2:] - sds[len(sds)/2:]))
        elif self.mode == 'mcs':
            self.env.reset()
            errors = []
            self.env.reset()
            for s_g in self.testcases:
                self.env.reset()
                
                context = self.env.get_current_context()
                in_dims = range(self.ag.conf.m_ndims, self.ag.conf.m_ndims + self.ag.conf.s_ndims)
                out_dims = range(self.ag.conf.m_ndims)
                m = self.ag.infer(in_dims, 
                                out_dims, 
                                np.array(context + list(s_g)))
                s = self.env.update(m, reset=False)
                errors.append(linalg.norm(s_g - s[len(context):]))
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

        self.ag.sensorimotor_model.mode = sm_mode
        return errors

    def plot_testcases(self, ax, dims, **kwargs_plot):
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        # test_array = array([hstack((m, s)) for m, s in self.tester.testcases])
        # test_array = test_array[:, dims]
        # ax.plot(*(test_array.T), **plot_specs)
        ax.plot(*(self.testcases.T), **plot_specs)
