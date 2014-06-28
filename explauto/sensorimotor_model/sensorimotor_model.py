from abc import ABCMeta, abstractmethod


class SensorimotorModel(object):
    """ This abstract class provides the common interface for sensorimotor models. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def infer(self, in_dims, out_dims):
        pass

    @abstractmethod
    def update(self, m, s):
        pass
