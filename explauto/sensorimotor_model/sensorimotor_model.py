from abc import ABCMeta, abstractmethod

from . import sensorimotor_models


class SensorimotorModel(object):
    """ This abstract class provides the common interface for sensorimotor models. """
    __metaclass__ = ABCMeta

    def __init__(self, conf):
        self.conf = conf

    @classmethod
    def from_configuration(cls, conf, sm_name, config_name='default'):
        sm_cls, sm_configs = sensorimotor_models[sm_name]
        return sm_cls(conf, **sm_configs[config_name])

    @abstractmethod
    def infer(self, in_dims, out_dims, x):
        """ Performs inference in the sensorimotor space.

        :param list in_dims: list of input dimensions. For example, use self.conf.m_dims to perform forward prediction (i.e. infer the expected sensory effect for a given input motor command) or self.conf.s_dims to perform inverse prediction (i.e. infer a motor command in order to reach an input sensory goal).

        :param list out_dims: list of output dimensions. For example, use self.conf.s_dims to perform forward prediction (i.e. infer the expected output sensory effect for a given input motor command) or self.conf.m_dims to perform inverse prediction (i.e. infer a output motor command in order to reach an input sensory goal).

        :param numpy.array x: value array for input dimensions. For example, if in_dims = self.conf.m_dims, x is the value of the motor configuration for which we want to predict a sensory effect.

        :returns: an array of size len(out_dims) containing the forward or inverse prediction

        .. note:: Although it is especially used to perform either forward or inverse predictions, Explauto's sensorimotor models are generally suitable to do all kind of general prediction from X (input) to Y (output), where X and Y are to distinct subspaces of the sensorimotor space.
        """
        pass

    @abstractmethod
    def update(self, m, s):
        """ Update the sensorimotor model given a new (m, s) pair, where m is a motor command and s is the corresponding observed sensory effect.
        """
        pass

    def forward_prediction(self, m):
        """ Compute the expected sensory effect of the motor command m. It is a shortcut for self.infer(self.conf.m_dims, self.conf.s_dims, m)
        """
        return self.infer(self.conf.m_dims, self.conf.s_dims, m)


    def inverse_prediction(self, s_g):
        """ Compute a motor command to reach the sensory goal s_g. It is a shortcut for self.infer(self.conf.s_dims, self.conf.m_dims, s_g)
        """
        return self.infer(self.conf.s_dims, self.conf.m_dims, s_g)
