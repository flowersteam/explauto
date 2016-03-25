import logging
import threading
import numpy as np

from copy import copy
from numpy import hstack

from ..exceptions import ExplautoEnvironmentUpdateError
from ..utils.observer import Observer
from ..evaluation import Evaluation

from ..agent import Agent
from .log import ExperimentLog
from ..environment import environments
from ..interest_model import interest_models
from ..sensorimotor_model import sensorimotor_models
from ..environment.context_environment import ContextEnvironment


logger = logging.getLogger(__name__)


class Experiment(Observer):
    def __init__(self, environment, agent, context_mode=None):
        """ This class is used to setup, run and log the results of an experiment.

            :param environment: an environment
            :type environment: :py:class:`~explauto.environment.environment.Environment`

            :param agent: an agent
            :type agent: :py:class:`~explauto.environment.agent.Agent`

            """
        Observer.__init__(self)

        self.env = environment
        self.ag = agent
        self.context_mode = context_mode
        # env.inds_in = inds_in
        # env.inds_out = inds_out
        # self.records = zeros((n_records, env.state.shape[0]))
        # self.i_rec = 0
        self.eval_at = []

        self.log = ExperimentLog(self.ag.conf, self.ag.expl_dims, self.ag.inf_dims)

        self.ag.subscribe('choice', self)
        self.ag.subscribe('inference', self)
        self.ag.subscribe('perception', self)
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

    def fast_forward(self, log):
        self.log = copy(log)
        for x, y, s in zip(*[log.logs[topic] for topic in ['choice', 'inference', 'perception']]):
            m, s_ag = self.ag.extract_ms(x, y)
            self.ag.sensorimotor_model.update(m, s)
            self.ag.interest_model.update(hstack((m, s_ag)), hstack((m, s)))

    def _run(self, n_iter):

        self._init()
        for _ in range(n_iter):
            self._step()
            if not self._running.is_set():
                break

        self._running.clear()

    def _init(self, current_step=0):
        self.current_step = current_step

    def _step(self):

        self.current_step += 1

        if self.current_step in self.eval_at and self.evaluation is not None:
            self.log.eval_errors.append(self.evaluation.evaluate())

        # Clear messages received from the evaluation
        self.notifications.queue.clear()

        try:
            if self.context_mode is None:
                m = self.ag.produce()
                env_state = self.env.update(m)
                self.ag.perceive(env_state)
            else:
                if self.context_mode.has_key('reset_iterations') and np.mod(self.current_step, self.context_mode['reset_iterations']) == 0:
                    self.env.reset()
                if self.context_mode["mode"] == 'mdmsds':
                    m = self.env.current_motor_position
                    s = self.env.current_sensori_position
                    mdm = self.ag.produce(list(m) + list(s))
                    sds = self.env.update(mdm, reset=False)
                    self.ag.perceive(sds, context=s)
                elif self.context_mode["mode"] == 'mcs':
                    context = self.env.get_current_context()
                    m = self.ag.produce(list(context))
                    s = self.env.update(m, reset=False)
                    self.ag.perceive(s, context=context)
                else:
                    raise NotImplementedError

        except ExplautoEnvironmentUpdateError:
            logger.warning('Environment update error at time %d with '
                           'motor command %s. '
                           'This iteration wont be used to update agent models',
                           self.current_step, m)

        self._update_logs()

    def _update_logs(self):
        while not self.notifications.empty():
            topic, msg = self.notifications.get()
            self.log.add(topic, msg)

    def evaluate_at(self, eval_at, testcases, mode=None):
        """ Sets the evaluation interation indices.

            :param list eval_at: iteration indices where an evaluation should be performed
            :param numpy.array testcases: testcases used for evaluation

        """
        self.eval_at = eval_at
        self.log.eval_at = eval_at

        if mode is None:
            if self.context_mode is None or (self.context_mode.has_key('choose_m') and self.context_mode['choose_m']):
                mode = 'inverse'
            else:
                mode = self.context_mode["mode"]
                

        self.evaluation = Evaluation(self.ag, self.env, testcases, mode=mode)
        for test in testcases:
            self.log.add('testcases', test)

    @classmethod
    def from_settings(cls, settings):
        """ Creates a :class:`~explauto.experiment.experiment.Experiment` object thanks to the given settings. """
        env_cls, env_configs, _ = environments[settings.environment]
        config = env_configs[settings.environment_config]

        if settings.context_mode is None:
            env = env_cls(**config)
        else:
            env = ContextEnvironment(env_cls, config, settings.context_mode)

        im_cls, im_configs = interest_models[settings.interest_model]
        sm_cls, sm_configs = sensorimotor_models[settings.sensorimotor_model]

        babbling = settings.babbling_mode
        if babbling not in ['goal', 'motor']:
            raise ValueError("babbling argument must be in ['goal', 'motor']")

        expl_dims = env.conf.m_dims if (babbling == 'motor') else env.conf.s_dims
        inf_dims = env.conf.s_dims if (babbling == 'motor') else env.conf.m_dims

        if settings.context_mode is None:
            agent = Agent.from_classes(im_cls, im_configs[settings.interest_model_config], expl_dims,
                          sm_cls, sm_configs[settings.sensorimotor_model_config], inf_dims,
                          env.conf.m_mins, env.conf.m_maxs, env.conf.s_mins, env.conf.s_maxs)
        else:
            agent = Agent.from_classes(im_cls, im_configs[settings.interest_model_config], expl_dims,
                          sm_cls, sm_configs[settings.sensorimotor_model_config], inf_dims,
                          env.conf.m_mins, env.conf.m_maxs, env.conf.s_mins, env.conf.s_maxs, context_mode=settings.context_mode)

        return cls(env, agent, settings.context_mode)
