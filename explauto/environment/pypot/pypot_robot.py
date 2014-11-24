import time
import numpy
import json
import os
from copy import copy

from ..environment import Environment
from ...utils import bounds_min_max

import pypot
pypot_path = os.path.join(os.path.dirname(os.path.abspath(pypot.__file__)), os.pardir)


class PypotEnvironment(Environment):
    """ Environment based on dynamixel based robot using pypot.

        This environment can be used to link explauto with pypot, a library allowing to control robot based on dynamixel motors. It uses an optitrack has the sensor. This could easily be changed by defining other pypot environments.

    """
    use_process = False

    def __init__(self,
                 robot_cls, robot_conf, motors, move_duration, tracked_obj,
                 m_mins, m_maxs, s_mins, s_maxs):
        """ :param get_pypot_robot: function returning the pypot robot used as the environment
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

        self.robot_explauto = robot_cls(**robot_conf)
        self.robot = self.robot_explauto.robot
        self.motors = [m.name for m in getattr(self.robot, motors)]
        self.move_duration = move_duration

        self.tracked_obj = tracked_obj
        self.robot.start_sync()

    def compute_motor_command(self, m_ag):
        """ Compute the motor command by restricting it to the bounds. """
        m_env = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        return m_env

    def compute_sensori_effect(self, m_env):
        """ Make the robot moves and retrieve the tracked object position. """
        cmd = numpy.rad2deg(m_env)  # Why???
        print "Why the fuck are we converting from rad 2 deg here???"
        pos = dict(zip(self.motors, cmd))
        self.robot.goto_position(pos, self.move_duration, wait=True)
        time.sleep(0.5)

        return self.robot_explauto.get_position(self.tracked_obj)



from numpy import deg2rad, array

# motor bounds for the left arm
l_m_mins = deg2rad(array([-15, 0, -90, -90]))
l_m_maxs = deg2rad(array([90, 90, 90, 0]))

# motor bounds for the right arm
r_m_mins = deg2rad(array([-15, -90, -90, -90]))
r_m_maxs = deg2rad(array([90, 0, 90, 0]))

# sensor bounds for the left arm
l_s_mins = array((-0.2, -0.1, 0.0))
l_s_maxs = array((0.4, 0.5, 0.6))

# sensor bounds for the right arm
r_s_mins = array((-0.2, -0.5, 0.0))
r_s_maxs = array((0.4, 0.1, 0.6))



class VrepRobot(object):
    def __init__(self, config_path, scene_path, host='127.0.0.1', port=19997, tracked_objects=['left_hand_tracker', 'right_hand_tracker']):
        from pypot.vrep import from_vrep

        with open(config_path) as cf:
            config = json.load(cf)
        self.robot = from_vrep(config, host, port, scene_path,
                          tracked_objects)
    def get_position(self, tracked_object):
        return getattr(self.robot, tracked_object).position

class PoppyRobot(object):
    def __init__(self, config_path):
        import pypot.robot

        from explauto.utils.tracker import OptiTracker
        from pypot.sensor.optibridge import OptiTrackClient
        opti = OptiTrackClient('193.50.110.222', 8989, ('l_hand', 'r_hand', 'wand'))
        opti.start()
        self.tracker = OptiTracker(opti)

        self.robot = pypot.robot.from_json(config_path)


    def get_position(self, tracked_object):
        return self.tracker.get_position(tracked_object)


def get_configuration(get_robot, tracker_cls, tracked_obj,
                      m_mins=l_m_mins, m_maxs=l_m_maxs,
                      s_mins=l_s_mins, s_maxs=l_s_maxs):
    pass


conf_poppy = {'robot_cls': PoppyRobot,
              'robot_conf': {'config_path':'../../poppy-software/poppytools/configuration/poppy_config.json'},
              'motors': 'l_arm',
              'move_duration': 1.0,
              'tracked_obj': 'left_hand_tracker',
              'm_mins': l_m_mins,
              'm_maxs': l_m_maxs,
              's_mins': l_s_mins,
              's_maxs': l_s_maxs}

conf_vrep = {'robot_cls': VrepRobot,
             'robot_conf': {'config_path': '../../poppy-software/poppytools/configuration/poppy_config.json',
                            'scene_path':os.path.join(pypot_path, 'samples/notebooks/poppy-sitting.ttt'),
                            'host':'127.0.0.1',
                            'port':19997,             'tracked_objects':['left_hand_tracker']},
              'motors': 'l_arm',
              'move_duration': 1.0,
              'tracked_obj': 'left_hand_tracker',
              'm_mins': l_m_mins,
              'm_maxs': l_m_maxs,
              's_mins': l_s_mins,
              's_maxs': l_s_maxs}

# conf_vrep_lying = copy(conf_vrep)
# conf_vrep_lying['robot_conf']['scene_path'] = os.path.join(pypot_path, 'samples/notebooks/poppy-lying.ttt'


configurations = {'poppy': conf_poppy, 'vrep':conf_vrep, 'default':conf_vrep}

environment = PypotEnvironment

testcases = None
