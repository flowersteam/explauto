import itertools

import pypot.robot
import pypot.primitive


class Arm(object):
    motors_name = ['base_pan', 'base_tilt_lower', 'base_tilt_upper',
                   'head_pan', 'head_tilt', 'head_roll', 'gripper']

    def __init__(self, json_file):
        self.robot = pypot.robot.from_json(json_file)

        for m in self.motors:
            setattr(self, m.name, m)

        self.imitation = None
        self.started = False

    def setup(self):
        if not self.started:
            self.robot.start_sync()
            self.started = True

        for m in self.motors:
            m.compliant = False

        self.goto_position(Arm.rest_position, 2)

    def record_position(self):
        return dict([(str(m.name), m.present_position) for m in self.motors])

    def goto_position(self, position, dt):
        self.robot.goto_position(position, dt)

    def imitate(self, imited):
        self.imitation = CopyPrimitive(self, imited, 50, max_speed=50)
        self.imitation.start()

    def stop_imitate(self):
        self.imitation.stop()

    def hold(self):
        self._tighten(0)

    def close(self):
        self._tighten(10)

    def release(self):
        self._tighten(50)

    @property
    def motors(self):
        return self.robot.motors

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


class CopyPrimitive(pypot.primitive.LoopPrimitive):
    def __init__(self, imitor, imited, freq, max_speed=50):
        pypot.primitive.LoopPrimitive.__init__(self, imitor.robot, freq)

        self.imitor = imitor
        self.imited = imited

        for m in self.imitor.motors:
            m.moving_speed = max_speed

    def update(self):
        pypot.primitive.LoopPrimitive.update(self)

        for name in self.imited.motors_name:
            m_imited = getattr(self.imited, name)
            m_imitor = getattr(self.imitor, name)

            m_imitor.goal_position = m_imited.present_position
