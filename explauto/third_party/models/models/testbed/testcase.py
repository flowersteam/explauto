"""Two function to generate set of test."""

import random
# import pandas

from explauto import ExplautoEnvironmentUpdateError


def uniform_motor_testcases(robot, n):
    """Generates n test uniformly distributed in the motor space"""
    tb = []
    for i in range(n):
        # order = pandas.Series([random.uniform(mi_min, mi_max)
        #                       for mi_min, mi_max in robot.m_bounds],
        #                      index = robot.m_feats)
        while True:
            try:
                order = [random.uniform(mi_min, mi_max)
                         for mi_min, mi_max in robot.m_bounds]
                effect = robot.execute_order(order)
                tb.append((order, effect))
                break
            except ExplautoEnvironmentUpdateError:
                pass
    return tb


class Lattice(object):
    """Select a subset of the provided observations approximately uniformly distributed
    in the sensory space."""

    def __init__(self, s_feats, observations, res = 10):
        """
        @param res           the resolution of the grid
        @param observations  the observations
        """
        self.s_feats       = s_feats
        self.res          = int(res)
        self.observations = observations
        self.grid         = {}
        self._populate()

    def _populate(self):
        """Fill the grid with observations"""
        self._compute_bounds()
        for i, obs in enumerate(self.observations):
            self._place_in_grid(obs)

    def _compute_bounds(self):
        """Compute the boundaries for each feature."""
        self.bounds = []
        for f in self.s_feats:
            minf, maxf = float('inf'), float('-inf')
            for order, effect in self.observations:
                minf = min(minf, effect[f])
                maxf = max(maxf, effect[f])
            self.bounds.append((minf, maxf))
        self.bounds = tuple(self.bounds)

    def _place_in_grid(self, obs):
        """Place an obs in the correct cell of the grid,
        creating it if necessary.
        """
        order, effect = obs
        coo = tuple(min(int((effect[f]-minf)/((maxf-minf)/self.res)), self.res-1)
                    for f, (minf, maxf) in zip(self.s_feats, self.bounds))
        # TODO : change obs only if nearer from center of coo.
        self.grid[coo] = obs


def uniform_sensor_testcases(robot, n):
    """Generates around n test uniformly distributed in the sensory space"""
    observations = uniform_motor_testcases(robot, 100*n)
    resolution = max(2, int((1.3*n)**(1.0/len(robot.s_feats))))
    lattice = Lattice(robot.s_feats, observations, res = resolution)
    return list(lattice.grid.values())
