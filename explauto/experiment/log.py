from collections import defaultdict
from numpy import array, mean, std


class ExperimentLog(object):
    def __init__(self):
        self._logs = defaultdict(list)
        self.counts = defaultdict(int)
        self.eval_errors = []

    @property
    def logs(self):
        return {key: array(val) for key, val in self._logs.iteritems()}

    def add(self, topic, message):
        self._logs[topic].append(message)
        self.counts[topic] += 1

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

    def plot_learning_curve(self, ax):
        if not hasattr(self, '_eval_at'):
            raise UserWarning('No evaluation available, '
                              'you need to specify the evaluate_at argument'
                              ' when constructing the experiment')

        avg_err = mean(array(self.eval_errors), axis=1)
        std_err = std(array(self.eval_errors), axis=1)
        ax.errorbar(self.eval_at, avg_err, yerr=std_err)

    # def density_plot(self, topic_dims, t=None,
