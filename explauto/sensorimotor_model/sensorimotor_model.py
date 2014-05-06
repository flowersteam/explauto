from abc import ABCMeta, abstractmethod


class SensorimotorModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def infer(self, in_dims, out_dims):
        pass

    @abstractmethod
    def update(self, m, s):
        pass
