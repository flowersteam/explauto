import time
import numpy

from .. import Environment
from ...utils import bounds_min_max
from ... import ExplautoEnvironmentUpdateError


class PypotEnvironment(Environment):
    """ Environment based on dynamixel based robot using pypot.

        This environment can be used to link explauto with pypot, a library allowing to control robot based on dynamixel motors. It uses an optitrack has the sensor. This could easily be changed by defining other pypot environments.

    """
    use_process = False

    def __init__(self,
                 pypot_robot, motors, move_duration,
                 optitrack_sensor, tracked_obj,
                 m_mins, m_maxs, s_mins, s_maxs):
        """ :param pypot_robot: robot used as the environment
            :type pypot_robot: :class:`~pypot.robot.robot.Robot`
            :param motors: list of motors used by the environment
            :type motors: list of :class:`~pypot.dynamixel.motor.DxlMotor`
            :param float move_duration: duration (in sec.) of each primitive motion
            :param optitrack_sensor: Optitrack used as the sensor by the :class:`~explauto.agent.agent.Agent`
            :type optitrack_sensor: :class:`~pypot.sensor.optitrack.OptiTrackClient`
            :param string tracked_obj: name of the object tracked by the optitrack
            :param numpy.array m_mins: minimum motor dims
            :param numpy.array m_maxs: maximum motor dims
            :param numpy.array s_mins: minimum sensor dims
            :param numpy.array s_maxs: maximum sensor dims

        """
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        self.readable = range(self.conf.ndims)

        self.robot = pypot_robot
        self.motors = [m.name for m in motors]
        self.move_duration = move_duration

        self.opti_get = lambda: optitrack_sensor.tracked_objects[tracked_obj].position

    def compute_motor_command(self, ag_state):
        """ Compute the motor command by restricting it to the bounds. """
        motor_cmd = ag_state
        return bounds_min_max(motor_cmd, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self):
        """ Make the robot moves and retrieve the tracked object position. """
        cmd = numpy.rad2deg(self.state[:self.conf.m_ndims])
        pos = dict(zip(self.motors, cmd))
        self.robot.goto_position(pos, self.move_duration, wait=True)
        time.sleep(0.5)

        try:
            return self.opti_get()
        except KeyError:
            raise ExplautoEnvironmentUpdateError
