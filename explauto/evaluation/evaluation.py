import random
from numpy import linalg, array, zeros, sum, min, max, transpose, where



class Evaluation(object):
    def __init__(self, log, ag, env, testcases):
        self.log = log
        self.ag = ag
        self.env = env
        self.testcases = testcases
        self.modes = self.log.config.eval_modes

        
    def evaluate(self):
        self.ag.eval_mode()
        if 'inverse' in self.modes:
            self.evaluate_comp()
        if 'explo' in self.modes:
            self.evaluate_explo()
        if 'explo_comp' in self.modes:
            self.evaluate_explo_comp()

    def test_inverse(self, s_g):
        m = self.ag.infer(self.ag.config.eval_dims, self.ag.conf.m_dims, s_g, pref = 'eval_').flatten()
        m_env = self.ag.motor_primitive(m)
        s_env = self.env.update(m_env, log=False)
        s_ = self.ag.sensory_primitive(s_env)
        s = self.ag.get_eval_dims(s_)
        return linalg.norm(s_g - s), s_

    def evaluate_comp(self):
        
        s_reached = []
        errors = []
        for s_g in self.testcases:
            e, s_ = self.test_inverse(s_g)
            s_reached.append(s_)
            errors.append(e)
            #print 'Evaluation', len(errors), ': s_goal = ', s_g, 's_reached = ', s_, 'L2 error = ', e, '\n'
        self.ag.learning_mode()
        #print s_reached
        self.log.eval_errors.append(errors)
        self.log.eval_reached.append(s_reached)
    

    def evaluate_explo(self):
        if self.log.logs.has_key('agentMS'):
                
            data_s = array([a[self.log.config.eval_explo_dims] for a in self.log.logs['agentMS']])
    #         print "explo eval data_s"
    #         for s in data_s:
    #             if s != [0.]:
    #                 print s
            eval_range = array([min(data_s, axis=0),
                               max(data_s, axis=0)])
            
            eps = self.log.config.eval_explo_eps
            grid_sizes = (eval_range[1,:] - eval_range[0,:]) / eps + 1
            grid_sizes = array(grid_sizes, dtype = int)
            grid = zeros(grid_sizes)
            
            for i in range(len(data_s)):
                idxs = array((data_s[i] - eval_range[0,:]) / eps, dtype=int)
                grid[tuple(idxs)] = grid[tuple(idxs)] + 1
                
            grid[grid > 1] = 1
            explo = sum(grid)
        else:
            explo = 0
        #print eval_range, eps, grid_sizes, grid
        self.log.explo.append(explo)
        print '[' + self.log.config.tag + '] ' + 'Exploration evaluation = ' + str(explo)
    
    
    def evaluate_explo_comp(self):
        
        data_s = array([a for a in self.log.logs['perception_mod'+'{}'.format(len(self.log.config.mids))]])
        print "data_s", data_s
        eval_range = array([min(data_s, axis=0),
                           max(data_s, axis=0)])
        
        eps = self.log.config.eval_explo_comp_eps
        grid_sizes = (eval_range[1,:] - eval_range[0,:]) / eps + 1
        grid_sizes = array(grid_sizes, dtype = int)
        grid = zeros(grid_sizes)
        
        for i in range(len(data_s)):
            idxs = array((data_s[i] - eval_range[0,:]) / eps, dtype=int)
            grid[tuple(idxs)] = grid[tuple(idxs)] + 1
            
        grid[grid > 1] = 1
        explo = sum(grid)
        self.log.explo_comp_explo.append(explo)
        print '[' + self.log.config.tag + '] ' + 'ExploComp evaluation =' + str(explo)
        
        if explo < len(self.testcases):
            #print "grid", grid
            to_test = list(transpose(array(where(grid))))
            random.shuffle(to_test)
            
            errors = []
            for idxs in to_test:
                s_g = (idxs + 0.5) * eps + eval_range[0,:]
                #print idxs, s_g
                e, s_reached = self.test_inverse(s_g)
                errors.append(e)
            max_dist = eps
            nb_comp = sum(array(errors)<max_dist)
            self.log.explo_comp.append(nb_comp)
            print "Evaluate explo comp", to_test, errors, max_dist, nb_comp
        

    def plot_testcases(self, ax, dims, **kwargs_plot):
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        # test_array = array([hstack((m, s)) for m, s in self.tester.testcases])
        # test_array = test_array[:, dims]
        # ax.plot(*(test_array.T), **plot_specs)
        ax.plot(*(self.testcases.T), **plot_specs)
