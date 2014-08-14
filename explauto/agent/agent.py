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

        :param class im_model_cls: a subclass of InterestedModel, as those registered in the interest_model package

        :param dict im_model_config: a configuration dict as those registered in the interest_model package

        :param list expl_dims: the sensorimotor dimensions where exploration is driven in the interest model

        :param list inf_dims: the output sensorimotor dimensions of the sensorimotor model (input being expl_dims)

        :param class sm_model_cls: a subclass of SensorimotorModel, as those registered in the sensorimotor_model package

        :param dict sensorimotor_model_config: a configuration dict as those registered in the sensorimotor_model package

        :param list m_mins, m_maxs, s_mins, s_max: lower and upper bounds of motor and sensory values on each dimension

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
        """ To be removed in future versions """
        if self.t > 0:
            self.perceive(env_state)
        self.t += 1
        return self.produce()

    def post_production(self):
        pass

    def pre_perception(self):
        pass

    def infer(self, expl_dims, inf_dims, x):
        """ Use the sensorimotor model to compute the expected value on inf_dims given that the value on expl_dims is x.

        .. note:: This corresponds to a prediction if expl_dims=self.conf.m_dims and inf_dims=self.conf.s_dims and to inverse prediction if expl_dims=self.conf.s_dims and inf_dims=self.conf.m_dims.
        """
        try:
            if self.n_bootstrap > 0:
                self.n_bootstrap -= 1
                raise ExplautoBootstrapError
            y = self.sensorimotor_model.infer(expl_dims,
                                              inf_dims,
                                              x.flatten())
        except ExplautoBootstrapError:
            logger.warning('Sensorimotor model not bootstrapped yet')
            y = rand_bounds(self.conf.bounds[:, inf_dims]).flatten()
        return y

    def produce(self):
        """ Exploration (see the `Explauto introduction <about.html>`__ for more detail):

        * Choose a value x on expl_dims according to the interest model
        * Infer a value y on inf_dims from x using the :meth:`~explauto.agent.agent.Agent.infer` method


        .. note:: This correspond to motor babbling if expl_dims=self.conf.m_dims and inf_dims=self.conf.s_dims and to  goal babbling if expl_dims=self.conf.s_dims and inf_dims=self.conf.m_dims.
        """

        try:
            self.x = self.interest_model.sample()
        except ExplautoBootstrapError:
            logger.warning('Interest model not bootstrapped yet')
            self.x = rand_bounds(self.conf.bounds[:, self.expl_dims]).flatten()

        self.y = self.infer(self.expl_dims, self.inf_dims, self.x)

        self.ms[self.expl_dims] = self.x
        self.ms[self.inf_dims] = self.y

        self.post_production()

        self.ms[self.conf.m_dims] = bounds_min_max(self.ms[self.conf.m_dims], self.conf.m_mins, self.conf.m_maxs)

        self.emit('choice', self.ms[self.expl_dims])
        self.emit('inference', self.ms[self.inf_dims])

        return self.ms[self.conf.m_dims]

    def perceive(self, ms):
        """ Learning (see the `Explauto introduction <about.html>`__ for more detail):

        * update the sensorimotor model with (m, s)
        * update the interest model with (x, y, m, s) (x, y are stored in self.ms in :meth:`~explauto.agent.agent.Agent.production`)
        """
        self.pre_perception()

        self.sensorimotor_model.update(self.ms[self.conf.m_dims], ms[self.conf.s_dims])
        self.interest_model.update(self.ms, ms)
        # self.t += 1
