import math, random
import collections
import treedict


import pandas

from . import robot

class RevJoint(object):
    """RevoluteJoint. Has one parent, and possibly several descendants"""

    def __init__(self, length = 1.0, limits = (-150.0, 150.0), orientation = 0.0,
                       feats = (None, None, None)):
        """
        @param length       the length of the body attached to the joint
        @param orientation  the initial orientation of the joints.
                            Limits are enforced relative to the origin.
        @param limits       the possible angle range.
        @param feats        tuple of available feats for the joint (x, y, angle)
                            None if not available
        """
        self.length  = length
        self.origin  = orientation
        self.limits  = tuple(limits)
        self.nodes   = []
        #assert len(feats) == 3
        self.feats = feats

    def forward_kin(self, pos_ref, a):
        """Compute the position of the end of the body attached to the joint
        @param x_ref, y_ref, a_ref  position and orientation of the end of the parent.
        @param a                    the angle requested (to be checked against limits)
        """
        a_min, a_max = self.limits
        a_legal = min(max(a_min, a), a_max)

        x_ref, y_ref, a_ref = pos_ref
        a_end = a_legal + self.origin + a_ref
        x_end = x_ref + self.length*math.cos(math.radians(a_end))
        y_end = y_ref + self.length*math.sin(math.radians(a_end))

        pos_end = x_end, y_end, a_end

        reading = {} if self.feats is None else {f : pos_end[i] for i, f in enumerate(self.feats) if f is not None}

        return pos_end, reading

    def add_node(self, node):
        self.nodes.append(node)


class MultiArm2D(object):
    """MultiArm class. Can simulate any revolute, non-cyclic kinematic tree.
    Order can be shuffled. Features cannot be shuffled yet.
    """

    def __init__(self):
        self.root = None
        self.joints = []
        self.readings = {}
        self.bounds = ()
        self.motormap = []

    def add_joint(self, parent, joint):
        if parent is None:
            assert len(self.joints) == 0, 'Tried to create a root in a non-empty multiarm'
            self.root = joint
        else:
            parent.add_node(joint)
        self.joints.append(joint)
        self.motormap.append(-len(self.motormap)-1)
        self._update_bounds()
        return joint

    def add_joint_randomly(self, joint):
        if self.root is None:
            return self.add_joint(None, joint)
        else:
            parent = random.choice(self.joints)
            return self.add_joint(parent, joint)

    def _shuffle_motors(self):
        """Shuffle the motors, that is, to which joints order values are applied.
        (should be done once at start, to randomize structure)
        """
        random.shuffle(self.motormap)

    def _reorder_order(self, order):
        return [order[i] for i in self.motormap]

    def forward_kin(self, order):
        """Compute the position of the end effector"""
        assert len(order) == len(self.joints), 'Exepcted an order with {} values, got {}'.format(len(self.joints), len(order))
        order_ed = self._reorder_order(order)
        self.readings = {}
        assert len(self._forward_spider(order_ed, (0.0, 0.0, 0.0), self.root)) == 0
        return pandas.Series(self.readings)

    def _forward_spider(self, ordertail, pos_ref, joint):
        a = ordertail[0]
        ordertail = ordertail[1:]
        pos_end, reading = joint.forward_kin(pos_ref, a)
        self.readings.update(reading)
        for j in joint.nodes:
            ordertail = self._forward_spider(ordertail, pos_end, j)
        return ordertail

    def _update_bounds(self):
        self.bounds = self._bounds_spider(self.root)

    def _bounds_spider(self, joint):
        bounds = [joint.limits]
        for j in joint.nodes:
            bounds += self._bounds_spider(j)
        return bounds

    def random_obs(self):
        """Return a random observation"""
        order = [random.uniform(lb, ub) for lb, ub in self.bounds]
        obs = pandas.Series(order, range(-len(self.bounds), 0))
        return obs.append(self.forward_kin(order))

defaultcfg = treedict.TreeDict()
defaultcfg.dim     = 6
defaultcfg.limits  = (-150, 150)
defaultcfg.lengths = 1.0
defaultcfg.s_feats = None
defaultcfg.m_feats = None

class KinematicArm2D(robot.Robot):
    """Interface for the kinematics of an arm"""

    def __init__(self, cfg = None, dim = 6):
        self.cfg = cfg
        if self.cfg is None:
            self.cfg = treedict.TreeDict()
            self.cfg.dim = dim
        self.cfg.update(defaultcfg, overwrite = False)

        self.dim      = self.cfg.dim
        self.m_feats  = tuple(range(-self.dim, 0)) if self.cfg.m_feats is None else self.cfg.m_feats
        self.m_bounds = tuple(self.cfg.limits for _ in range(self.dim))
        self.s_feats  = (0, 1) if self.cfg.s_feats is None else self.cfg.s_feats
        self.name = 'KinematicArm2D'
        self._init_robot(self.cfg.lengths, self.cfg.limits)

    def _init_robot(self, lengths, limits):
        self._multiarm = MultiArm2D()

        # create self.lengths
        if isinstance(lengths, collections.Iterable):
            assert len(lengths) == len(self.m_feats)
            self.lengths = lengths
        else:
            self.lengths = [lengths for mf in self.m_feats]
        self.same_lengths = all(l == self.lengths[0] for l in self.lengths)
        self.total_length = sum(self.lengths)

        # create self.limits
        if isinstance(limits[0], collections.Iterable):
            assert len(limits) == len(self.m_feats)
            assert all(len(l_i) == 2 for l_i in limits)
            self.m_bounds = limits
        else:
            assert len(limits) == 2
            self.m_bounds = [limits for mf in self.m_feats]
        self.same_limits = all(l == self.m_bounds[0] for l in self.m_bounds)

        j = None
        for i in range(len(self.m_feats)):
            feats = None if i < len(self.m_feats)-1 else (0, 1, None) # x,y for the tip only
            j = self._multiarm.add_joint(j, RevJoint(length = self.lengths[i], limits = self.m_bounds[i], orientation = 0.0, feats = feats))

    def execute_order(self, order):
        order_t = self._pre_x(order)
        effect = self._multiarm.forward_kin(order)
        return self._post_y(effect, order)

    def __repr__(self):
        return "KinematicArm2D(dim = {}, lengths = {}, m_bounds = {})".format(
               self.dim, self.lengths, self.m_bounds)
