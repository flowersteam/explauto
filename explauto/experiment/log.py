from collections import defaultdict
from numpy import array, mean, std


class ExperimentLog(object):
    def __init__(self, conf, expl_dims, inf_dims):
        self._logs = defaultdict(list)
        self.counts = defaultdict(int)
        self.eval_errors = []
        self.conf = conf
        self.expl_dims = expl_dims
        self.inf_dims = inf_dims

    @property
    def logs(self):
        return {key: array(val) for key, val in self._logs.iteritems()}

    def add(self, topic, message):
        self._logs[topic].append(message)
        self.counts[topic] += 1

    def pack(self, topic_dims, t):
        """ Packs selected logs into a numpy array
            :param list topic_dims: list of (topic, dims) tuples, where topic is a string and dims a list dimensions to be packed for that topic
            :param int t: time indexes to be packed
        """

        data = []
        for topic, dims in topic_dims:
            for d in dims:
                data.append(self.logs[topic][t, d])
        return array(data).T

    def axes_limits(self, topic_dims):
        bounds = []
        for topic, dims in topic_dims:
            if topic == 'motor':
                bounds.extend(list(self.conf.m_bounds[:, dims].T.flatten()))
            elif topic == 'sensori':
                bounds.extend(list(self.conf.s_bounds[:, dims].T.flatten()))
            elif topic == 'choice':
                bounds.extend(list(self.conf.bounds[:, [self.expl_dims[d]
                                                        for d in dims]].T.flatten()))
            elif topic == 'inference':
                bounds.extend(list(self.conf.bounds[:, [self.inf_dims[d]
                                                        for d in dims]].T.flatten()))
            else:
                raise ValueError("Only valid for 'motor', 'sensori', 'choice' and 'inference' topics")
        return bounds

    def scatter_plot(self, ax, topic_dims, t=None, ms_limits=True, **kwargs_plot):
        """ 2D or 3D scatter plot
            :param axes ax: matplotlib axes (use Axes3D if 3D data)
            :param tuple topic_dims: list of (topic, dims) tuples, where topic is a string and dims is a list of dimensions to be plotted for that topic.
            :param int t: time indexes to be plotted            :param dict kwargs_plot: argument to be passed to matplotlib's plot function, e.g. the style of the plotted points 'or'
            :param bool ms_limits: if set to True, automatically set axes boundaries to the sensorimotor boundaries (default: True)
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
        if ms_limits:
            ax.axis(self.axes_limits(topic_dims))

    def plot_learning_curve(self, ax):
        if not hasattr(self, 'eval_at'):
            raise UserWarning('No evaluation available, '
                              'you need to specify the evaluate_at argument'
                              ' when constructing the experiment')

        avg_err = mean(array(self.eval_errors), axis=1)
        std_err = std(array(self.eval_errors), axis=1)
        ax.errorbar(self.eval_at, avg_err, yerr=std_err)



    # def density_plot(self, topic_dims, t=None,
