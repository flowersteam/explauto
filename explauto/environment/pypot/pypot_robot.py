import time
import numpy

from .. import Environment
from ...utils import bounds_min_max
from ... import ExplautoEnvironmentUpdateError


class PypotEnvironment(Environment):
    def __init__(self,
                 pypot_robot, motors, move_duration,
                 optitrack_sensor, tracked_obj,
                 m_mins, m_maxs, s_mins, s_maxs):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.readable = range(self.conf.ndims)

        self.robot = pypot_robot
        self.motors = [m.name for m in motors]
        self.move_duration = move_duration

        self.opti_getter = lambda: optitrack_sensor.tracked_objects[tracked_obj].position

    def compute_motor_command(self, ag_state):
        motor_cmd = ag_state
        return bounds_min_max(motor_cmd, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self):
        cmd = numpy.rad2deg(self.state[:self.conf.m_ndims])
        pos = dict(zip(self.motors, cmd))
        self.robot.goto_position(pos, self.move_duration, wait=True)
        time.sleep(0.5)

        try:
            return self.opti_getter()
        except KeyError:
            raise ExplautoEnvironmentUpdateError
