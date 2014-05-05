from abc import ABCMeta, abstractmethod


class InterestModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, i_dims):
        self.i_dims = i_dims

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, xy, ms):
        pass
