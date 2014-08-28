from abc import ABCMeta, abstractmethod

from ..exceptions import ExplautoEnvironmentUpdateError


class Tracker(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_position(self, tracked_object):
        pass


class OptiTracker(Tracker):
    def __init__(self, optitrack):
        self.opti = optitrack

    def get_position(self, tracked_object):
        try:
            return self.opti.tracked_objects[tracked_object].position

        except KeyError:
            raise ExplautoEnvironmentUpdateError
