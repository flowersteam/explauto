import time
import numpy

from ..environment import Environment
from ...utils import bounds_min_max


class PypotEnvironment(Environment):
    """ Environment based on dynamixel based robot using pypot.

        This environment can be used to link explauto with pypot, a library allowing to control robot based on dynamixel motors. It uses an optitrack has the sensor. This could easily be changed by defining other pypot environments.

    """
    use_process = False

    def __init__(self,
                 pypot_robot, motors, move_duration,
                 tracker, tracked_obj,
                 m_mins, m_maxs, s_mins, s_maxs):
        """ :param pypot_robot: robot used as the environment
            :type pypot_robot: :class:`~pypot.robot.robot.Robot`
            :param motors: list of motors used by the environment
            :type motors: list of :class:`~pypot.dynamixel.motor.DxlMotor`
            :param float move_duration: duration (in sec.) of each primitive motion
            :param tracker: tracker used as the sensor by the :class:`~explauto.agent.agent.Agent`
            :type tracker: :class:`~explauto.utils.tracker.Tracker`
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

        self.tracker = tracker
        self.tracked_obj = tracked_obj

    def compute_motor_command(self, m_ag):
        """ Compute the motor command by restricting it to the bounds. """
        m_env = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        return m_env

    def compute_sensori_effect(self, m_env):
        """ Make the robot moves and retrieve the tracked object position. """
        cmd = numpy.rad2deg(m_env)
        pos = dict(zip(self.motors, cmd))
        self.robot.goto_position(pos, self.move_duration, wait=True)
        time.sleep(0.5)

        return self.tracker.get_position(self.tracked_obj)
