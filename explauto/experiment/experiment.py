import logging
import threading

from .. import ExplautoEnvironmentUpdateError
from ..utils.observer import Observer
from ..evaluation import Evaluation

from ..agent import Agent
from .log import ExperimentLog
from ..environment import environments
from ..interest_model import interest_models
from ..sensorimotor_model import sensorimotor_models


logger = logging.getLogger(__name__)


class Experiment(Observer):
    def __init__(self, environment, agent):
        """ This class is used to setup, run and log the results of an experiment.

            :param environment: an environment
            :type environment: :py:class:`~explauto.environment.environment.Environment`

            :param agent: an agent
            :type agent: :py:class:`~explauto.environment.agent.Agent`

            """
        Observer.__init__(self)

        self.env = environment
        self.ag = agent
        # env.inds_in = inds_in
        # env.inds_out = inds_out
        # self.records = zeros((n_records, env.state.shape[0]))
        # self.i_rec = 0
        self.eval_at = []

        self.log = ExperimentLog(self.env.conf, self.ag.expl_dims, self.ag.inf_dims)

        self.ag.subscribe('choice', self)
        self.ag.subscribe('inference', self)
        self.env.subscribe('motor', self)
        self.env.subscribe('sensori', self)

        self._running = threading.Event()

    def run(self, n_iter=-1, bg=False):
        """ Run the experiment.

            :param int n_iter: Number of run iterations, by default will run until the last evaluation step.
            :param bool bg: whether to run in background (using a Thread)

        """
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
        """ Wait for the end of the run of the experiment. """
        self._t.join()

    def stop(self):
        """ Stop the experiment. """
        self._running.clear()

    def _run(self, n_iter):
        for t in range(n_iter):
            if t in self.eval_at and self.evaluation is not None:
                self.log.eval_errors.append(self.evaluation.evaluate())

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

            self._update_logs()

            if not self._running.is_set():
                break

        self._running.clear()

    def _update_logs(self):
        while not self.notifications.empty():
            topic, msg = self.notifications.get()
            self.log.add(topic, msg)

    def evaluate_at(self, eval_at, testcases):
        """ Sets the evaluation interation indices.

            :param list eval_at: iteration indices where an evaluation should be performed
            :param numpy.array testcases: testcases used for evaluation

        """
        self.eval_at = eval_at
        self.log.eval_at = eval_at

        self.evaluation = Evaluation(self.ag, self.env, testcases)
        for test in testcases:
            self.log.add('testcases', test)

    @classmethod
    def from_settings(cls, settings):
        """ Creates a :class:`~explauto.experiment.experiment.Experiment` object thanks to the given settings. """
        env_cls, env_configs, _ = environments[settings.environment]
        config = env_configs[settings.environment_config]

        env = env_cls(**config)

        im_cls, im_configs = interest_models[settings.interest_model]
        sm_cls, sm_configs = sensorimotor_models[settings.sensorimotor_model]

        babbling = settings.babbling_mode
        if babbling not in ['goal', 'motor']:
            raise ValueError("babbling argument must be in ['goal', 'motor']")

        expl_dims = env.conf.m_dims if (babbling == 'motor') else env.conf.s_dims
        inf_dims = env.conf.s_dims if (babbling == 'motor') else env.conf.m_dims

        agent = Agent(im_cls, im_configs[settings.interest_model_config], expl_dims,
                      sm_cls, sm_configs[settings.sensorimotor_model_config], inf_dims,
                      env.conf.m_mins, env.conf.m_maxs, env.conf.s_mins, env.conf.s_maxs)

        return cls(env, agent)
