from numpy import zeros
class Experiment(object):
    def __init__(self, env, ag, inds_in, inds_out, n_records = 100000):
        self.env = env
        self.ag = ag
        env.inds_in = inds_in
        env.inds_out = inds_out
        self.records = zeros((n_records, env.state.shape[0]))
        self.i_rec = 0

    def run(self, n_iter = 1):
        for _ in range(n_iter):
            self.env.write(self.ag.produce())
            self.env.next_state()
            self.ag.perceive(self.env.read())
            self.records[self.i_rec,:] = self.env.state
            self.i_rec += 1
            

        
