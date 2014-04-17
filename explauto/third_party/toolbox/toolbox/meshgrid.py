import numbers
import random

import numpy as np

class MeshCell(object):

    def __init__(self, coo, bounds = None):
        self.coo      = coo
        self.bounds   = bounds
        self.elements = []
        self.metadata = []

    def add(self, e, metadata = None):
        self.elements.append((e, metadata))

    def __len__(self):
        return len(self.elements)

    def draw(self, replace = True):
        idx = random.randint(0, len(self.elements) - 1)
        if replace:
            return self.elements[idx]
        else:
            return self.elements.pop(idx)


class MeshGrid(object):

    def __init__(self, bounds, res):
        all(isinstance(e, numbers.Number) for e in bounds)
        self.bounds = bounds
        if isinstance(res, int):
            self.res = len(bounds)*[res]
        else:
            assert len(res) == len(bounds) and all(isinstance(e, int) for e in res)
            self.res = res

        self.dim = len(bounds)
        self._cells = {}
        self._cells[None] = MeshCell(None) # a cell for everything not inside the bounds
        self._size = 0
        self._nonempty_cells = []

    def _coo(self, p):
        assert len(p) == self.dim
        coo = []
        for pi, (si_min, si_max), res_i in zip(p, self.bounds, self.res):
            coo.append(int((pi - si_min)/(si_max - si_min)*res_i))
            if pi == si_max:
                coo[-1] == res_i - 1
            if si_min > pi or si_max < pi:
                return None
        return tuple(coo)

    def __len__(self):
        return self._size

    def add(self, p, metadata = None):
        assert len(p) == self.dim
        self._size += 1
        coo = self._coo(p)
        if not coo in self._cells:
            self._cells[coo] = MeshCell(coo)
        cell = self._cells[coo]
        cell.add(p, metadata)
        if len(cell) == 1:
            self._nonempty_cells.append(cell)

    def draw(self, replace = True):
        """Draw uniformly between existing (non-empty) cells"""
        try:
            idx = random.randint(0, len(self._nonempty_cells) - 1)
        except ValueError:
            raise ValueError("can't draw from an empty meshgrid")
        e, md = self._nonempty_cells[idx].draw(replace = replace)
        if not replace and len(self._nonempty_cells[idx]) == 0:
            self._nonempty_cells.pop(idx)
        return e, md
