from ..dataset import Dataset

class ForwardModel(object):
    """Class describing the ForwardModel interface"""

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """Construct a Nearest Neighbor forward model from an existing dataset."""
        m = cls(dataset.dim_x, dataset.dim_y, **kwargs)
        m.dataset = dataset
        return m

    @classmethod
    def from_robot(cls, robot, **kwargs):
        """Construct a Nearest Neighbor forward model from an existing dataset."""
        m = cls(len(robot.m_feats), len(robot.s_feats), **kwargs)
        return m

    def __init__(self, dim_x, dim_y, **kwargs):
        """Create the forward model

        @param dim_x    the input dimension
        @param dim_y    the output dimension
        """
        self.dim_x    = dim_x
        self.dim_y    = dim_y
        self.dataset  = Dataset(dim_x, dim_y)
        self.conf     = kwargs

    def reset(self):
        self.dataset.reset()

    def size(self):
        return self.dataset.size

    def add_xy(self, x, y):
        """Add an observation to the forward model

        @param x  an array of float of length dim_in
        @param y  an array of float of length dim_out
        """
        self.dataset.add_xy(x, y)

    def get_x(self, index):
        return self.dataset.get_x(index)

    def get_y(self, index):
        return self.dataset.get_y(index)

    def get_xy(self, index):
        return self.dataset.get_xy(index)

    def predict_y(self, xq, **kwargs):
        """Provide an prediction of xq in the output space

        @param xq  an array of float of length dim_x
        """
        raise NotImplementedError

    def config(self):
        """Return a string with the configuration"""
        return ", ".join('%s:%s' % (key, value) for key, value in self.conf.items())


import random

class RandomForwardModel(ForwardModel):
    """Random Forward Model"""

    name = 'Rnd'
    desc = 'Random'

    def predict_y(self, x):
        idx = random.randint(0, self.size()-1)
        return self.dataset.get_y(idx)
