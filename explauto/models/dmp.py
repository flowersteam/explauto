from numpy import zeros, ones, array
from copy import copy

from pydmps.dmp_rhythmic import DMPs_rhythmic
from pydmps.dmp_discrete import DMPs_discrete

class MotorPrimitive(object):
    def __init__(self, conf):
        pass

    def trajectory(self, params):
        pass

class DmpPrimitive(object):
    def __init__(self, dmps, bfs, used=None, default=None, type='discrete', timesteps=100):
        self.used = ones(dmps * (bfs + 2), dtype=bool) if used is None else array(used, dtype=bool)
        self.default = zeros(dmps*(bfs + 2)) if default is None else array(default)
        self.motor = copy(self.default)
        self.n_dmps = dmps
        self.n_bfs = bfs
        
        if type == 'discrete':
            self.dmp = DMPs_discrete(dmps=dmps, bfs=bfs, dt=2./timesteps)
        elif type =='rythmic':
            #dt = 6.28 / timesteps
            self.dmp = DMPs_rhythmic(dmps=dmps, bfs=bfs)#, dt=dt)
        else:
            raise ValueError('Invalid type specified. Valid choices \
                                 are discrete or rhythmic.')
        #self.dmp.cs.run_time *= run_time
        #self.dmp.timesteps *= run_time
    def trajectory(self, m, n_times=1):
        #print "trajectory. n_dmps :", self.n_dmps, 'n_bfs', self.n_bfs, 'run time', self.dmp.cs.run_time, 'time steps', self.dmp.timesteps
        self.dmp.cs.run_time *= n_times
        self.dmp.timesteps *= n_times
        self.motor[self.used] = m
        self.dmp.y0 = self.motor[:self.dmp.dmps]
        self.dmp.goal = self.motor[-self.dmp.dmps:]
        self.dmp.w = self.motor[self.dmp.dmps:-self.dmp.dmps].reshape(self.dmp.dmps, self.dmp.bfs)
        y, dy, ddy = self.dmp.rollout()
        self.dmp.cs.run_time /= n_times
        self.dmp.timesteps /= n_times
        return y
