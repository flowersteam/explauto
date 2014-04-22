from numpy import zeros, array
from collections import defaultdict

from ..utils.observer import Observer


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

        self.ag.subscribe('choice', self)
        self.ag.subscribe('inference', self)
        self.env.subscribe('motor', self)
        self.env.subscribe('sensori', self)

    def run(self, n_iter=1):
        for _ in range(n_iter):
            # if i_rec in evaluate_at:
            #     self.evaluation.evaluate(self.env, self.ag, self.testset, self.
            self.env.update(self.ag.next_state(self.env.read()))
            # self.env.write(self.ag.produce())
            # self.env.next_state()
            # self.ag.perceive(self.env.read())
            self.records[self.i_rec, :] = self.env.state
            self.i_rec += 1

            self.update_logs()

    def update_logs(self):
        while not self.notifications.empty():
            topic, msg = self.notifications.get()
            self._logs[topic].append(msg)

    @property
    def logs(self):
        return {key: array(val) for key, val in self._logs.iteritems()}
