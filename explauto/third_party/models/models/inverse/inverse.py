
import random
import numpy as np

from ....toolbox import toolbox
# import models.forward

relax = 0.0 # no relaxation

class InverseModel(object):

    @classmethod
    def from_dataset(cls, dataset, sigma, **kwargs):
        """dim_xstruct a optimized inverse model from an existing dataset."""
        raise NotImplementedError

    @classmethod
    def from_forward(cls, fmodel, **kwargs):
        """Construst an inverse model from a forward model and constraints.
        """
        im = cls(fmodel.dim_x, fmodel.dim_y, **kwargs)
        im.fmodel = fmodel
        return im

    def __init__(self, dim_x, dim_y, **kwargs):
        """Construst an inverse model from a dimensions and constraints set
        Default to a LWR model for the forward model.
        """
        self.k = kwargs.get('k', 3*dim_y)
        self.conf = kwargs

    def infer_x(self, y):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        """
        assert len(y) == self.fmodel.dim_y, "Wrong dimension for y. Expected %i, got %i" % (self.fmodel.dim_y, len(y))
        self.y_desired = np.array(y)

    def _random_x(self):
        """If the database is empty, generate a random vector."""
        return (tuple(random.random() for _ in range(self.fmodel.dim_x)),)


    def _guess_x(self, y_desired, **kwargs):
        """Choose the relevant neighborhood to feed the inverse model, based
        on the minimum spread of the corresponding neighborhood in S.
        for each (x, y) with y neighbor of y_desired,
            1. find the neighborhood of x, (xi, yi)_k.
            2. compute the standart deviation of the error between yi and y_desired.
            3. select the neighborhood of minimum standart deviation

        TODO : Implement another method taking the spread in M too.
        """
        k = kwargs.get('k', self.k)
        dists, indexes = self.fmodel.dataset.nn_y(y_desired, k = k)
        min_std, min_xi = float('inf'), None
        for i in indexes:
            xi = self.fmodel.dataset.get_x(i)
            dists_xi, indexes_xi = self.fmodel.dataset.nn_x(xi, k = k)
            std_xi = np.std([toolbox.dist(self.fmodel.dataset.get_y(j), y_desired) for j in indexes_xi])
            if std_xi < min_std:
                min_std, min_xi = std_xi, xi
        return [min_xi]


    def _guess_x_simple(self, y_desired, **kwargs):
        """Provide an initial guesses for a probable x from y"""
        dists, indexes = self.fmodel.dataset.nn_y(y_desired, n = 10)

        return [self.fmodel.get_x(i) for i in indexes]

    def add_xy(self, x, y):
        self.add_x(x)
        self.fmodel.add_xy(x, y)

    def add_x(self, x):
        """If the model needs to dynamically update the motor extermum, this
        is the method you're looking for."""
        pass

    def config(self):
        """Return a string with the configuration"""
        return ", ".join('%s:%s' % (key, value) for key, value in self.conf.items())

class RandomInverseModel(InverseModel):
    """Random Inverse Model"""

    name = 'Rnd'
    desc = 'Random'


    def infer_x(self, y_desired):
        """Infer probable x from input y

        @param y  the desired output for infered x.
        """
        InverseModel.infer_x(y_desired)
        if self.fmodel.size() == 0:
            return self._random_x()
        else:
            idx = random.randint(0, self.fmodel.size()-1)
            return self.dataset.get_x(idx)
