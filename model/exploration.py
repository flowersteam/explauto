import numpy as np

class Agent(object):
    def __init__(self, m_dims, s_dims, env, sm_model, i_model, competence):
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
        self.ms = np.zeros((len(self.ms_dims), 1))
        self.env = env
        self.sm_model = sm_model
        self.i_model = i_model
        self.competence = competence
        m, s = self.env.execute(np.zeros((len(self.m_dims), 1)))
        self.sm_model.update(m, s)

    def explore(self, in_dims, out_dims):
        x = self.i_model.sample()
        y = self.sm_model.infer(in_dims, out_dims, x, mode='explore')
        self.ms[in_dims] = x
        self.ms[out_dims] = y
        m = self.ms[self.m_dims].reshape(len(self.m_dims), 1)
        m, s = self.env.execute(m)
        comp = self.competence(self.ms[self.s_dims], s)
        self.sm_model.update(m, s)
        self.i_model.update(x, comp)
        return x, y, m, s, comp
