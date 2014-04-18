from numpy import zeros
from ..utils.observer import Observable

class Experiment(Observable):
    def __init__(self, env, ag, n_records=100000, evaluate_at=[]):
        self.env = env
        self.ag = ag
        #env.inds_in = inds_in
        #env.inds_out = inds_out
        self.records = zeros((n_records, env.state.shape[0]))
        self.i_rec = 0
        self.evaluate_at = evaluate_at

    def run(self, n_iter = 1):
        for _ in range(n_iter):
            #if i_rec in evaluate_at:
                #self.evaluation.evaluate(self.env, self.ag, self.testset, self. 
            self.env.next_state(self.ag.next_state(self.env.read()))
            #self.env.write(self.ag.produce())
            #self.env.next_state()
            #self.ag.perceive(self.env.read())
            self.records[self.i_rec,:] = self.env.state
            self.i_rec += 1
            

        
