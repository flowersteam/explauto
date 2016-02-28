from ... import Environment
from ...utils import bounds_min_max


class PoppyEnvironment(Environment):
    """ Explauto environment for Poppy robots.

        It can be used with all existing poppy creatures:
            * Humanoid
            * Torso
            * ErgoJr

        It can be used with real or simulated robots (using V-REP for instance).

    """
    def __init__(self,
                 poppy_robot,
                 motors, move_duration,
                 tracker, tracked_obj,
                 m_mins, m_maxs, s_mins, s_maxs):
        """"
        :param poppy_robot: PoppyCreature instance (it can be a real or a simulated robot)
        :param list motors: list of motors used - it can directly be a motor alias, e.g.m poppy.l_arm
        :param float move_duration: duration of the motor commands
        :param tracker: Tracker used to determine the tracked_obj position - when using a robot simulated with V-REP the robot itself can be the tracker.
        :param str tracked_obj: name of the object to track
        :param numpy.array m_mins: minimum motor dims
        :param numpy.array m_maxs: maximum motor dims
        :param numpy.array s_mins: minimum sensor dims
        :param numpy.array s_maxs: maximum sensor dims

        """
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.robot = poppy_robot
        self.motors = motors
        self.move_duration = move_duration

        self.tracker = tracker
        self.tracked_obj = tracked_obj

    def compute_motor_command(self, m_ag):
        """ Compute the motor command by restricting it to the bounds. """
        m_env = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        return m_env

    def compute_sensori_effect(self, m_env):
        """ Move the robot motors and retrieve the tracked object position. """
        pos = {m.name: pos for m, pos in zip(self.motors, m_env)}
        self.robot.goto_position(pos, self.move_duration, wait=True)

        # This allows to actually apply a motor command
        # Without having a tracker
        if self.tracker is not None:
            return self.tracker.get_object_position(self.tracked_obj)

    def reset(self):
        """ Resets simulation and does nothing when using a real robot. """
        if self.robot.simulated:
            self.robot.reset_simulation()
