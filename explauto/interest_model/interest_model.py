from abc import ABCMeta, abstractmethod


class InterestModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, expl_dims):
        self.expl_dims = expl_dims

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, xy, ms):
        pass
