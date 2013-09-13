import numpy

import pypot.robot
import pypot.sensor.optitrack as opti
from pypot.kinematics import Link, Chain, transl, trotx, translation_from_transf

from scipy.linalg import norm
from numpy import pi

base_pos = dict([('left_shoulder_pan', 8.599999999999994),
                 ('left_shoulder_twist', 87.24),
                 ('left_shoulder_lift', -67.89),
                 ('left_elbow_twist', 13.64),
                 ('left_elbow_lift', 88.42),
                 ('left_wrist', 76.61),
                 ('left_hand', -4.090000000000003)])

numpy.set_printoptions(precision=3, suppress=True)

def opti2torso(pos):
    x, y, z = pos
    return numpy.array((-x, -z, -y))

class Torso(object):
    def __init__(self, dummy_robot=False, os='win', opti_addr='127.0.0.1'):
        if not dummy_robot:
            self.robot = pypot.robot.from_configuration('../torso-{}.xml'.format(os))
            self.robot.start_sync()

        self.opti = opti.OptiTrackClient(opti_addr, 3883, ('left_hand', 'right_hand', 'goal'))
        self.opti.start()

        l1 = Link(0, 0, 0, pi/2)
        l2 = Link(0, 0.11, 0.0158, -pi/2)
        l3 = Link(-pi, 0.062, 0, -pi/2)
        l4 = Link(0, 0, 0, pi/2)
        l5 = Link(pi/2, 0.15133, 0, -pi/2)
        l6 = Link(0, 0, 0, pi/2)
        l7 = Link(pi/2, 0.13141, 0, pi/2)

        base = transl(-0.07, 0.234, 0.1) * trotx(-pi/2)
        tool = transl(-0.035, -0.028, -0.016)

        self.left_arm_chain = Chain((l1, l2, l3, l4, l5, l6, l7), base, tool)

        l1 = Link(0, 0, 0, pi/2)
        l2 = Link(0, 0.11, 0.0158, -pi/2)
        l3 = Link(-pi, -0.062, 0, -pi/2)
        l4 = Link(0, 0, 0, pi/2)
        l5 = Link(pi/2, -0.15133, 0, -pi/2)
        l6 = Link(0, 0, 0, pi/2)
        l7 = Link(pi/2, -0.13141, 0, pi/2)

        base = transl(-0.07, 0.148, 0.1) * trotx(-pi/2)
        tool = transl(-0.024, 0.016, -0.012)

        self.right_arm_chain = Chain((l1, l2, l3, l4, l5, l6, l7), base, tool)

    @property
    def left_hand_position(self):
        """ The left_hand_position property."""
        return self._get_obj_position('left_hand')

    @property
    def left_q(self):
        """ The left_q property."""
        angles = [m.present_position for m in [self.robot.left_trunk_lift] + self.robot.left_arm[:-1]]
        return numpy.array(map(numpy.deg2rad, angles))

    @property
    def left_hand_kinematic_position(self):
        """ The left_hand_kinematic_position property."""
        return translation_from_transf(self.left_arm_chain.forward_kinematics(self.left_q)[0])

    @property
    def right_hand_position(self):
        """ The right_hand_position property."""
        return self._get_obj_position('right_hand')

    @property
    def right_q(self):
        """ The left_q property."""
        angles = [m.present_position for m in [self.robot.right_trunk_lift] + self.robot.right_arm[:-1]]
        return numpy.array(map(numpy.deg2rad, angles))

    @property
    def right_hand_kinematic_position(self):
        """ The right_hand_kinematic_position property."""
        return translation_from_transf(self.right_arm_chain.forward_kinematics(self.right_q)[0])

    @property
    def goal_position(self):
        """ The goal_position property."""
        return self._get_obj_position('goal')

    @property
    def goal_distance(self):
        """ The goal_distance property."""
        return norm(self.left_hand_position - self.goal_position)

    @property
    def goal_alignement(self):
        p1 = self.left_hand_position
        p2 = self.goal_position

        return norm(numpy.array((p1[0], p1[2]) - numpy.array((p2[0], p2[2]))))

    def _get_obj_position(self, obj_name):
        # Warn: we assume that we can always track the object !
        p = self.opti.tracked_objects[obj_name].position
        return opti2torso(p)

    def goto(self, q, side, duration, wait):
        q = numpy.insert(q, 0, q[0])
        links = self.robot.trunk_lift + getattr(self.robot, '{}_arm'.format(side))[:-1]
        d = dict(zip([m.name for m in links], q))
        self.robot.goto_position(d, duration, wait)

    def look(self, pos, wait=True):
        self.robot.goto_position({'head_pan': 45 * pos[0],
                                  'head_tilt': 25 + 25 * pos[1]}, 1, wait)

if __name__ == '__main__':
    torso = Torso()
