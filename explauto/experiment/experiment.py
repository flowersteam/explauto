import logging
import threading

from numpy import array, mean, std
from collections import defaultdict

from .. import ExplautoEnvironmentUpdateError
from ..utils.observer import Observer
from ..evaluation import Evaluation
from ..utils import rand_bounds

# from ..utils import density_image
from ..agent import Agent
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

        self._logs = defaultdict(list)
        self.counts = defaultdict(int)

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
                self.eval_errors.append(self.evaluation.evaluate())

            # To clear messages received from the evaluation
            self.notifications.queue.clear()
            m = self.ag.produce()
            try:
                self.env.update(m)
                self.ag.perceive(self.env.state)
            except ExplautoEnvironmentUpdateError:
                logger.warning('Environment update error at time %d with '
                               'motor command %s. '
                               'This iteration wont be used to update agent models', t, m)

            # self.records[self.i_rec, :] = self.env.state
            # self.i_rec += 1

            self.update_logs()

            if not self._running.is_set():
                break

        self._running.clear()

    def update_logs(self):
        while not self.notifications.empty():
            topic, msg = self.notifications.get()
            self._logs[topic].append(msg)
            self.counts[topic] += 1

    @property
    def logs(self):
        return {key: array(val) for key, val in self._logs.iteritems()}

    def pack(self, topic_dims, t):
        """ Packs selected logs into a numpy array
            :param list topic_dims: list of (topic, dims) tuples, where topic is a string and dims a list dimensions to be plotted for each topic
            :param int t: time indexes to be plotted
        """

        data = []
        for topic, dims in topic_dims:
            for d in dims:
                data.append(self.logs[topic][t, d])
        return array(data).T

    def evaluate_at(self, eval_at, evaluation=None):
        self.eval_at = eval_at
        if evaluation is None:
            self.evaluation = Evaluation(self.ag, self.env)
        else:
            self.evaluation = evaluation
        self.eval_errors = []

    def scatter_plot(self, ax, topic_dims, t=None, **kwargs_plot):
        """ 2D or 3D scatter plot
            :param dict topic_dims: dictionary of the form {topic : dims, ...}, where topic is a string and dims is a list of dimensions to be plotted for that topic.
            :param int t: time indexes to be plotted
            :param axes ax: matplotlib axes (use Axes3D if 3D data)
            :param dict kwargs_plot: argument to be passed to matplotlib's plot function, e.g. the style of the plotted points 'or'
        """
        plot_specs = {'marker': 'o', 'linestyle': 'None'}
        plot_specs.update(kwargs_plot)
        t_bound = float('inf')
        if t is None:
            for topic, _ in topic_dims:
                t_bound = min(t_bound, self.counts[topic])
            t = range(t_bound)
        data = self.pack(topic_dims, t)
        # ax.plot(data[:, 0], data[:, 1], style)
        ax.plot(*(data.T), **plot_specs)

    def plot_learning_curve(self, ax):
        if not self.evaluate_at:
            print 'no evaluation available, you need to specify the evaluate_at argument when constructing the experiment'
            return
        avg_err = mean(array(self.eval_errors), axis=1)
        std_err = std(array(self.eval_errors), axis=1)
        ax.errorbar(self.eval_at, avg_err, yerr=std_err)

    # def density_plot(self, topic_dims, t=None,

    @classmethod
    def from_settings(cls, environment, babbling, interest_model, sensorimotor_model,
                      environment_config='default',
                      interest_model_config='default',
                      sensorimotor_model_config='default'):

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
