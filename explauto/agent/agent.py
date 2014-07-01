import logging
import numpy as np


from ..utils.config import make_configuration
from ..utils.observer import Observable
from ..utils import rand_bounds, bounds_min_max
from .. import ExplautoBootstrapError

logger = logging.getLogger(__name__)


class Agent(Observable):
    def __init__(self,
                 im_model_cls, im_model_config, expl_dims,
                 sm_model_cls, sm_model_config, inf_dims,
                 m_mins, m_maxs, s_mins, s_maxs, n_bootstrap=0):
        """Initialize agent class
        Keyword arguments:
        :param class im_model_cls: a subclass of InterestedModel, as those registered in the interest_model package
        :param dict im_model_config: a configuration dict as those registered in the interest_model package
        :param expl_dims: the sensorimotor dimensions where exploration is driven in the interest model
        :param inf_dims: the output sensorimotor dimensions of the sensorimotor model (input being expl_dims)
        :param class sm_model_cls: a subclass of SensorimotorModel, as those registered in the sensorimotor_model package
        :param dict sensorimotor_model_config: a configuration dict as those registered in the sensorimotor_model package
        :params list m_mins: lower bound of motor values on each motor dimension (upper for m_max)
        :params list m_mins: lower bound of sensory values on each sensory dimension (lower for m_max)
        """
        Observable.__init__(self)

        self.conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)

        self.ms = np.zeros(self.conf.ndims)
        self.expl_dims = expl_dims
        self.inf_dims = inf_dims

        self.sensorimotor_model = sm_model_cls(self.conf, **sm_model_config)
        self.interest_model = im_model_cls(self.conf, self.expl_dims,
                                           **im_model_config)

        # self.competence = competence
        self.t = 0
        self.n_bootstrap = n_bootstrap
        self.state = np.zeros(self.conf.ndims)

    def next_state(self, env_state):
        if self.t > 0:
            self.perceive(env_state)
        self.t += 1
        return self.produce()

    def post_production(self):
        pass

    def pre_perception(self):
        pass

    def infer(self, expl_dims, inf_dims, x):
        try:
            if self.n_bootstrap > 0:
                self.n_bootstrap -= 1
                raise ExplautoBootstrapError
            y = self.sensorimotor_model.infer(expl_dims,
                                              inf_dims,
                                              x.flatten())
        except ExplautoBootstrapError:
            logger.warning('Sensorimotor model not bootstrapped yet, or Agent still in bootstraping phase')
            y = rand_bounds(self.conf.bounds[:, inf_dims]).flatten()
        return y

    def produce(self):
        # if self.to_bootstrap:
        #     return self.bootstrap()

        self.x = self.interest_model.sample()
        self.y = self.infer(self.expl_dims, self.inf_dims, self.x)

        self.ms[self.expl_dims] = self.x
        self.ms[self.inf_dims] = self.y

        self.post_production()

        self.ms[self.conf.m_dims] = bounds_min_max(self.ms[self.conf.m_dims], self.conf.m_mins, self.conf.m_maxs)

        self.emit('choice', self.ms[self.expl_dims])
        self.emit('inference', self.ms[self.inf_dims])

        return self.ms[self.conf.m_dims]

    def perceive(self, ms):
        # Todo: put competence function in i_model.py and call it from i_model
        self.pre_perception()

        # self.comp = self.competence(self.ms[self.s_dims], ms[self.s_dims])
        # self.comps[self.t] = self.comp

        self.sensorimotor_model.update(self.ms[self.conf.m_dims], ms[self.conf.s_dims])
        self.interest_model.update(self.ms, ms)
        # self.t += 1
