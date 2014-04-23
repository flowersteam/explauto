import time
import numpy
import pypot.robot

from .. import Environment
from ...utils import bounds_min_max
from ...explauto import ExplautoEnvironmentUpdateError


class PypotEnvironment(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 robot_config, motors, move_duration,
                 opti_track, tracked_obj):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.robot = pypot.robot.from_config(robot_config)
        self.robot.start_sync()

        self.motors = motors  # [m.name for m in motors]
        self.move_duration = move_duration

        self.opti_track = opti_track
        self.tracked_obj = tracked_obj

        self.readable = range(self.conf.ndims)

    def compute_motor_command(self, ag_state):
        motor_cmd = ag_state
        return bounds_min_max(motor_cmd, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self):
        cmd = numpy.rad2deg(self.state[:self.conf.m_ndims])
        pos = dict(zip(self.motors, cmd))

        self.robot.goto_position(pos, self.move_duration, wait=True)
        time.sleep(0.5)
        try:
            return self.opti_track.tracked_objects[self.tracked_obj].position
        except KeyError:
            raise ExplautoEnvironmentUpdateError
