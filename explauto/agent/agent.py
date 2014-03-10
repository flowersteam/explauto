import numpy as np

class Agent(object):
    def __init__(self, m_dims, s_dims, i_dims, inf_dims, sm_model, i_model):
        """Initialize agent class
        Keyword arguments:
        m_dims -- the indices of motor values
        s_dims -- the indices of sensory values
        sm_model -- the sensorimotor model
        i_model -- the interest model
        """
        self.m_dims = m_dims
        self.s_dims = s_dims
        self.ms_dims = self.m_dims + self.s_dims
        self.ms = np.zeros(len(self.ms_dims))
        self.i_dims = i_dims
        self.inf_dims = inf_dims
        self.sm_model = sm_model
        self.i_model = i_model
        #self.competence = competence
        self.to_bootstrap = True
        #self.choices = np.zeros((10000, len(i_dims)))
        #self.comps = np.zeros((10000, 1))
        self.t = 0
        self.state = np.zeros(len(self.ms_dims))

    #def bootstrap(self):
        #self.ms[self.m_dims] =np.zeros(len(self.m_dims))
        ##self.ms[self.s_dims] = self.i_model.bounds[0,:].reshape(-1,1)
        #self.x =np.array(self.i_model.bounds[0,:].reshape(-1,1))
        #self.to_bootstrap = False
        #return self.ms[self.m_dims].T

        #m, s = self.env.execute(np.zeros((len(self.m_dims), 1)))
        #self.ms = np.vstack((m, s))
        #self.sm_model.update(m, s)
        #self.i_model.update(self.ms[i_dims,:], self.competence(self.i_model.bounds[0,:].reshape(-1,1), self.i_model.bounds[1,:].reshape(-1,1)))

    def next_state(self, env_state):
        if self.t > 0:
            self.perceive(env_state)
        self.t += 1
        return self.produce()

    def post_production(self):
        pass

    def pre_perception(self):
        pass

    def produce(self):
        #if self.to_bootstrap:
            #return self.bootstrap()
        self.x = self.i_model.sample()
        self.y = self.sm_model.infer(self.i_dims, self.inf_dims, self.x)
        self.ms[self.i_dims] = self.x
        self.ms[self.inf_dims] = self.y
        #self.choices[self.t,:] = self.x
        self.post_production()
        return self.ms[self.m_dims]        

    def perceive(self, ms): 
        # Todo: put competence function in i_model.py and call it from i_model
        self.pre_perception()
        #self.comp = self.competence(self.ms[self.s_dims], ms[self.s_dims])
        #self.comps[self.t] = self.comp
        self.sm_model.update(ms[self.m_dims], ms[self.s_dims])
        self.i_model.update(self.ms, ms)
        #self.t += 1
        
    #def explore(self, in_dims, out_dims):
        #x = self.i_model.sample()
        #y = self.sm_model.infer(in_dims, out_dims, x)
        #self.ms[in_dims] = x
        #self.ms[out_dims] = y
        #m = self.ms[self.m_dims].reshape(len(self.m_dims), 1)
        #m, s = self.env.execute(m)
        #comp = self.competence(self.ms[self.s_dims], s)
        #self.sm_model.update(m, s)
        #self.i_model.update(x, comp)
        #return x, y, m, s, comp


