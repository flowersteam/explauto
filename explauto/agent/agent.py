import numpy as np


from ..utils.config import Configuration


class Agent(object):
    def __init__(self,
                 im_model_cls, im_model_config, expl_dims,
                 sm_model_cls, sm_model_config, inf_dims,
                 conf_dict):
        """Initialize agent class
        Keyword arguments:
        m_dims -- the indices of motor values
        s_dims -- the indices of sensory values
        sm_model -- the sensorimotor model
        i_model -- the interest model
        """
        conf = Configuration(conf_dict)
        for k, v in conf.iteritems():
            setattr(self, k, v)

        # self.m_dims = conf.m_dims
        # self.s_dims = conf.s_dims
        # self.ms_dims = self.m_dims + self.s_dims
        self.ms = np.zeros(self.ndims)
        # self.expl_dims = conf.expl_dims
        # self.inf_dims = conf.inf_dims

        self.expl_dims = expl_dims
        self.inf_dims = inf_dims

        # self.sensorimotor_model = sm_model
        # self.interest_model = i_model

        self.sensorimotor_model = sm_model_cls(conf, **sm_model_config)
        self.interest_model = im_model_cls(self.expl_dims, conf['bounds'], **im_model_config)

        # self.competence = competence
        self.to_bootstrap = True
        # self.choices = np.zeros((10000, len(i_dims)))
        # self.comps = np.zeros((10000, 1))
        self.t = 0
        self.state = np.zeros(self.ndims)

    # def bootstrap(self):
    #     self.ms[self.m_dims] =np.zeros(len(self.m_dims))
    #     #self.ms[self.s_dims] = self.interest_model.bounds[0,:].reshape(-1,1)
    #     self.x =np.array(self.interest_model.bounds[0,:].reshape(-1,1))
    #     self.to_bootstrap = False
    #     return self.ms[self.m_dims].T
    #
    #     m, s = self.env.execute(np.zeros((len(self.m_dims), 1)))
    #     self.ms = np.vstack((m, s))
    #     self.sensorimotor_model.update(m, s)
    #     self.interest_model.update(self.ms[i_dims,:],
    #                         self.competence(self.interest_model.bounds[0,:].reshape(-1,1),
    #                         self.interest_model.bounds[1,:].reshape(-1,1)))

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
        # if self.to_bootstrap:
        #     return self.bootstrap()

        self.x = self.interest_model.sample()
        self.y = self.sensorimotor_model.infer(self.expl_dims, self.inf_dims, self.x)

        self.ms[self.expl_dims] = self.x
        self.ms[self.inf_dims] = self.y

        # self.choices[self.t,:] = self.x

        self.post_production()

        return self.ms[self.m_dims]

    def perceive(self, ms):
        # Todo: put competence function in i_model.py and call it from i_model
        self.pre_perception()

        # self.comp = self.competence(self.ms[self.s_dims], ms[self.s_dims])
        # self.comps[self.t] = self.comp

        self.sensorimotor_model.update(ms[self.m_dims], ms[self.s_dims])
        self.interest_model.update(self.ms, ms)
        # self.t += 1

    # def explore(self, in_dims, out_dims):
    #     x = self.interest_model.sample()
    #     y = self.sensorimotor_model.infer(in_dims, out_dims, x)
    #     self.ms[in_dims] = x
    #     self.ms[out_dims] = y
    #     m = self.ms[self.m_dims].reshape(len(self.m_dims), 1)
    #     m, s = self.env.execute(m)
    #     comp = self.competence(self.ms[self.s_dims], s)
    #     self.sensorimotor_model.update(m, s)
    #     self.interest_model.update(x, comp)
    #     return x, y, m, s, comp
