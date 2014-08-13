from numpy import zeros
from copy import copy

from pydmps.dmp_rhythmic import DMPs_rhythmic
from pydmps.dmp_discrete import DMPs_discrete

class MotorPrimitive(object):
    def __init__(self, conf):
        pass

    def trajectory(self, params):
        pass

class DmpPrimitive(object):
    def __init__(self, dmps, bfs,used=None, default=None, type='discrete'):
        self.used = ones(dmp*(bfs + 2)) if used is None else used
        self.default = zeros(dmps*(bfs + 2)) if default is None else default
        self.motor = copy(self.default)
        if type == 'discrete':
            self.dmp = DMPs_discrete(dmps=dmps, bfs=bfs)
        elif type =='rythmic':
            self.dmp = DMPs_rhythmic(dmps=dmps, bfs=bfs)
        else:
            raise ValueError('Invalid type specified. Valid choices \
                                 are discrete or rhythmic.')
    def trajectory(self, m):
        self.motor[self.used] = m
        # print self.motor
        self.dmp.y0 = self.motor[:self.dmp.dmps]
        self.dmp.goal = self.motor[-self.dmp.dmps:]
        self.dmp.w = self.motor[self.dmp.dmps:-self.dmp.dmps].reshape(self.dmp.dmps, self.dmp.bfs)
        y, dy, ddy = self.dmp.rollout()
        return y
