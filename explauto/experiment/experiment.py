import logging
import threading

from .. import ExplautoEnvironmentUpdateError
from ..utils.observer import Observer
from ..evaluation import Evaluation
from ..utils import rand_bounds

from ..agent import Agent
from .log import ExperimentLog
from ..environment import environments
from ..interest_model import interest_models
from ..sensorimotor_model import sensorimotor_models


logger = logging.getLogger(__name__)


class Experiment(Observer):
    def __init__(self, environment, agent):
        """ This class is used to setup, run and plot the results of an experiment.

            :param environment: set a handler that will receive the different errors
            :type environment: :py:class:`~explauto.environment.environment.Environment`

            :param agent: set a handler that will receive the different errors
            :type agent: :py:class:`~explauto.environment.environment.Environment`

            """
        Observer.__init__(self)

        self.env = environment
        self.ag = agent
        # env.inds_in = inds_in
        # env.inds_out = inds_out
        # self.records = zeros((n_records, env.state.shape[0]))
        # self.i_rec = 0
        self.eval_at = []

        self.logs = ExperimentLog()

        self.ag.subscribe('choice', self)
        self.ag.subscribe('inference', self)
        self.env.subscribe('motor', self)
        self.env.subscribe('sensori', self)

        self._running = threading.Event()

    def bootstrap(self, n):
        while n > 0:
            m = rand_bounds(self.ag.conf.m_bounds)[0]

            try:
                self.env.update(m)
                self.ag.perceive(self.env.state)
                n -= 1

            except ExplautoEnvironmentUpdateError:
                pass

    def run(self, n_iter=-1, bg=False):
        if n_iter == -1:
            if not self.eval_at:
                raise ValueError('Set n_iter or define evaluate_at.')

            n_iter = self.eval_at[-1] + 1

        self._running.set()

        if bg:
            self._t = threading.Thread(target=lambda: self._run(n_iter))
            self._t.start()
        else:
            self._run(n_iter)

    def wait(self):
        self._t.join()

    def stop(self):
        self._running.clear()

    def _run(self, n_iter):
        for t in range(n_iter):
            if t in self.eval_at and self.evaluation is not None:
                self.logs.eval_errors.append(self.evaluation.evaluate())

            # Clear messages received from the evaluation
            self.notifications.queue.clear()

            m = self.ag.produce()
            try:
                self.env.update(m)
                self.ag.perceive(self.env.state)
            except ExplautoEnvironmentUpdateError:
                logger.warning('Environment update error at time %d with '
                               'motor command %s. '
                               'This iteration wont be used to update agent models',
                               t, m)

            # self.records[self.i_rec, :] = self.env.state
            # self.i_rec += 1

            self.update_logs()

            if not self._running.is_set():
                break

        self._running.clear()

    def update_logs(self):
        while not self.notifications.empty():
            topic, msg = self.notifications.get()
            self.logs.add(topic, msg)

    def evaluate_at(self, eval_at, testcases=None, evaluation=None):
        self.eval_at = eval_at
        self.logs.eval_at = eval_at

        if evaluation is None:
            self.evaluation = Evaluation(self.ag, self.env)
        else:
            self.evaluation = evaluation

        if testcases is not None:
            self.evaluation.tester.testcases = testcases

    @classmethod
    def from_settings(cls, environment, babbling, interest_model, sensorimotor_model,
                      environment_config='default',
                      interest_model_config='default',
                      sensorimotor_model_config='default'):
        """ Creates a :class:`~explauto.experiment.experiment.Experiment` object thanks to the given settings.

        :param str environment: Name of the environment (see :data:`~explauto.environment.environments`)
        :param str babbling: Babbling mode ('motor' or 'goal')
        :param str interest_model: Name of the interest model (see :data:`~explauto.interest_model.interest_models`)
        :param str sensorimotor_model: Name of the sensorimotor model (see :data:`~explauto.sensorimotor_model.sensorimotor_models`)
        :param str environment_config: Name of the environment config (default: 'default')
        :param str interest_model_config: Name of the interest model config (default: 'default')
        :param str sensorimotor_model_config: Name of the sensorimotor model config (default: 'default')

        This method automatically instantiate the :class:`~explauto.environment.environment.Environment` and the :class:`~explauto.agent.agent.Agent` with their respective configurations.

        .. note:: The name of the environment (resp interest, sensorimotor model) should be registred in the environments (resp. interest_models, sensorimotor_models) dictionnary. Similarly, the name of the configuration should correspond to one of the  registred configurations.

        """
        env_cls, env_configs = environments[environment]
        config = env_configs[environment_config]

        env = env_cls(**config)

        im_cls, im_configs = interest_models[interest_model]
        sm_cls, sm_configs = sensorimotor_models[sensorimotor_model]

        if babbling not in ['goal', 'motor']:
            raise ValueError("babbling argument must be in ['goal', 'motor']")

        expl_dims = env.conf.m_dims if (babbling == 'motor') else env.conf.s_dims
        inf_dims = env.conf.s_dims if (babbling == 'motor') else env.conf.m_dims

        agent = Agent(im_cls, im_configs[interest_model_config], expl_dims,
                      sm_cls, sm_configs[sensorimotor_model_config], inf_dims,
                      env.conf.m_mins, env.conf.m_maxs, env.conf.s_mins, env.conf.s_maxs)

        return cls(env, agent)
