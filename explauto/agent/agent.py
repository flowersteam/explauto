import logging
import numpy as np


from ..utils.config import make_configuration
from ..utils.observer import Observable
from ..utils import rand_bounds, bounds_min_max
from ..exceptions import ExplautoBootstrapError

logger = logging.getLogger(__name__)


class Agent(Observable):
    def __init__(self, conf, sm_model, im_model, n_bootstrap=0):
        Observable.__init__(self)
        self.conf = conf
        self.ms = np.zeros(self.conf.ndims)
        self.expl_dims = im_model.expl_dims
        self.inf_dims = sorted(list(set(conf.dims) - set(self.expl_dims)))

        self.sensorimotor_model = sm_model
        self.interest_model = im_model

        # self.competence = competence
        self.t = 0
        self.n_bootstrap = n_bootstrap

    @classmethod
    def from_classes(cls,
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

        conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)

        # self.ms = np.zeros(self.conf.ndims)
        # self.expl_dims = expl_dims
        # self.inf_dims = inf_dims

        sm_model = sm_model_cls(conf, **sm_model_config)
        im_model = im_model_cls(conf, expl_dims,
                                           **im_model_config)

        return cls(conf, sm_model, im_model)
        # # self.competence = competence
        # self.t = 0
        # self.n_bootstrap = n_bootstrap
        # self.state = np.zeros(self.conf.ndims)


    def choose(self):
        """ Returns a point chosen by the interest model
        """
        try:
            x = self.interest_model.sample()
        except ExplautoBootstrapError:
            logger.warning('Interest model not bootstrapped yet')
            x = rand_bounds(self.conf.bounds[:, self.expl_dims]).flatten()
        #print "interest model choose y : ", x
        self.emit('choice' + '_' + self.mid, x)
        return x


    def infer(self, expl_dims, inf_dims, x, pref = ''):
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
        
        
        self.emit(pref + 'inference' + '_' + self.mid, y)
        
        return y


    def extract_ms(self, x, y):
        """ Returns the motor and sensory parts from a point in the exploration
        space (expl_dims) and a point in the inference space (inf_dims).
        """
        ms = np.zeros(self.conf.ndims)
        ms[self.expl_dims] = x
        ms[self.inf_dims] = y
        return ms[self.conf.m_dims], ms[self.conf.s_dims]

    def motor_primitive(self, m):
        """ Prepare the movement from a command m. To be overridded in order to generate more complex movement (tutorial to come). This version simply bounds the command.
        """
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def sensory_primitive(self, s):
        """ Extract features from a sensory effect s. To be overridded in order to process more complex feature extraction (tutorial to come). This version simply bounds the sensory effect.
        """
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)

    def produce(self):
        """ Exploration (see the `Explauto introduction <about.html>`__ for more detail):

        * Choose a value x on expl_dims according to the interest model
        * Infer a value y on inf_dims from x using the :meth:`~explauto.agent.agent.Agent.infer` method
        * Extract the motor m and sensory s parts of x, y
        * generate a movement from m

        :returns: the generated movement

        .. note:: This correspond to motor babbling if expl_dims=self.conf.m_dims and inf_dims=self.conf.s_dims and to  goal babbling if expl_dims=self.conf.s_dims and inf_dims=self.conf.m_dims.
        """

        self.x = self.choose()
        self.y = self.infer(self.expl_dims, self.inf_dims, self.x)

        self.m, self.s = self.extract_ms(self.x, self.y)

        movement = self.motor_primitive(self.m)

        self.emit('choice', self.x)
        self.emit('inference', self.y)
        self.emit('movement', movement)

        return movement


    def perceive(self, s_):
        """ Learning (see the `Explauto introduction <about.html>`__ for more detail):

        * update the sensorimotor model with (m, s)
        * update the interest model with (x, y, m, s)
          (x, y are stored in self.ms in :meth:`~explauto.agent.agent.Agent.production`)
        """
        s = self.sensory_primitive(s_)
        self.emit('perception', s)
        self.sensorimotor_model.update(self.m, s)
        self.interest_model.update(np.hstack((self.m, self.s)), np.hstack((self.m, s)))
        self.t += 1
