from abc import ABCMeta, abstractmethod

from . import interest_models

class InterestModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, expl_dims):
        self.expl_dims = expl_dims

    @classmethod
    def from_configuration(cls, conf, expl_dims, im_name, config_name='default'):
        im_cls, im_configs = interest_models[im_name]
        return im_cls(conf, expl_dims, **im_configs[config_name])

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, xy, ms):
        pass
