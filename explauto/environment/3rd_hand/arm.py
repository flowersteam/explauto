import itertools
import numpy

import pypot.robot

from pypot.primitive import LoopPrimitive


class Arm(object):
    motors_name = ['base_pan', 'base_tilt_lower', 'base_tilt_upper',
                   'head_pan', 'head_tilt', 'head_roll', 'gripper']

    def __init__(self, json_file, opti):
        self.robot = pypot.robot.from_json(json_file)
        self.robot.start_sync()

        for m in self.motors:
            setattr(self, m.name, m)

        self.imitation = None

        self.opti = opti

    def setup(self):
        self.compliant = False
        self.goto_position(Arm.rest_position, 2)

    def record_position(self):
        return dict([(str(m.name), m.present_position) for m in self.motors])

    def goto_position(self, position, dt, wait=False):
        self.robot.goto_position(position, dt, wait)

    def imitate(self, imited):
        self.imitation = CopyPrimitive(self, imited, 50, max_speed=50)
        self.imitation.start()

    def stop_imitate(self):
        self.imitation.stop()

    def hold(self):
        self._tighten(0)

    def close_gripper(self):
        self._tighten(10)

    def open_gripper(self):
        self._tighten(50)

    @property
    def compliant(self):
        return self.robot.compliant

    @compliant.setter
    def compliant(self, value):
        self.robot.compliant = value

    @property
    def motors(self):
        return self.robot.motors

    @property
    def joints(self):
        return numpy.array([m.present_position for m in self.motors])

    def _get_opti_obj(self, obj_name):
        obj = self.opti.recent_tracked_objects[obj_name]
        return numpy.hstack((obj.position, obj.orientation))

    @property
    def tip(self):
        return self._get_opti_obj('gripper')

    @property
    def magicwand(self):
        return self._get_opti_obj('wand')

    def _tighten(self, pos, max_speed=50):
        self.gripper.moving_speed = max_speed
        self.gripper.goal_position = pos

    init_position = dict(zip(motors_name, itertools.repeat(0)))
    rest_position = {
        'base_pan': 0.0,
        'base_tilt_lower': 75.0,
        'base_tilt_upper': -50.0,
        'gripper': 0.0,
        'head_pan': 0.0,
        'head_roll': 0.0,
        'head_tilt': 20.0,
        'gripper': 10.0}


class CopyPrimitive(LoopPrimitive):
    def __init__(self, imitor, imited, freq, max_speed=50):
        LoopPrimitive.__init__(self, imitor.robot, freq)

        self.imitor = imitor
        self.imited = imited
        self.max_speed = max_speed

    def setup(self):
        for m in self.imitor.motors:
            m.moving_speed = self.max_speed

    def update(self):
        for name in self.imited.motors_name:
            m_imited = getattr(self.imited, name)
            m_imitor = getattr(self.imitor, name)

            m_imitor.goal_position = m_imited.present_position


class RecordPrimitive(LoopPrimitive):
    def __init__(self, arm, freq):
        LoopPrimitive.__init__(self, arm.robot, freq)
        self.arm = arm

    def setup(self):
        self._data = []

    def update(self):
        try:
            self._data.append(numpy.hstack((self.arm.joints, self.arm.tip)))
        except KeyError:
            pass

    @property
    def data(self):
        return numpy.array(self._data)
