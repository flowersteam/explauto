import logging

from numpy import zeros, array
from collections import defaultdict

from .. import ExplautoEnvironmentUpdateError
from ..utils.observer import Observer
from ..utils import rand_bounds

logger = logging.getLogger(__name__)


class Experiment(Observer):
    def __init__(self, env, ag, n_records=100000, evaluate_at=[]):
        Observer.__init__(self)

        self.env = env
        self.ag = ag
        # env.inds_in = inds_in
        # env.inds_out = inds_out
        self.records = zeros((n_records, env.state.shape[0]))
        self.i_rec = 0
        self.evaluate_at = evaluate_at

        self._logs = defaultdict(list)
        self.counts = defaultdict(int)

        self.ag.subscribe('choice', self)
        self.ag.subscribe('inference', self)
        self.env.subscribe('motor', self)
        self.env.subscribe('sensori', self)

    def bootstrap(self, n):
        while n > 0:
            m = rand_bounds(self.ag.conf.m_bounds)[0]

            try:
                self.env.update(m)
                self.ag.perceive(self.env.state)
                n -= 1

            except ExplautoEnvironmentUpdateError:
                pass

    def run(self, n_iter=1):
        for t in range(n_iter):
            # if i_rec in evaluate_at:
            #     self.evaluation.evaluate(self.env, self.ag, self.testset, self.
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
