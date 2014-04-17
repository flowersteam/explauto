"""A class to wrap up matplotlib code used to display the competence map of a (2D) model."""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import toolbox

def c_fun_exp(x):
    """Default function for competence"""
    return math.exp(-x)

margin = 10
margin = 10


from ..dataset import Dataset

class HeatMap(object):
    """Plot a heat map from a set of (3D) points."""

    @classmethod
    def from_xyc(cls, xyc_array, **kwargs):
        """Initialize the heat map from two arrays
        @param xyc_array  3d points (x, y, c) with the c the value of each point.
        """
        return cls(Dataset.from_xy([(x,y) for x,y,c in xyc_array], 
                                   [(c,) for x,y,c in xyc_array],), **kwargs)

    @classmethod
    def from_xy_c(cls, xy_array, c_array, **kwargs):
        """Initialize the heat map from two arrays
        @param xy_array  2d points for the spatial data
        @param c_array   added value (usually competence or interest)
        """
        c_points = tuple((c,) for c in c_array)
        return cls(Dataset.from_xy(xy_array, c_points), **kwargs)

    def __init__(self, dataset, res = 50, subplot = None, extent = None , reverse = False, **kwargs):
        """Initialize the heat map with a dataset
        @param dataset  dataset with, in x, 2d points for the spatial data
                        and y the value of each point (as arrays of length 1)
        This is the dataset that the heatmap use internally and initialization
        will just link it.
        It means that adding points to the dataset will update the heatmap.
        """
        assert dataset.dim_x == 2 and dataset.dim_y == 1, (
                 "Error, the dimension of the data is incorrect.\n"
                 +"Expected 3x2 and got %ix%i" % (dataset.dim_x, dataset.dim_y))
        self.res = res
        self.dataset = dataset
        self.subplot = subplot
        self.extent  = extent
        self.reversed = reverse

    def _meshgrid(self, res = None):
        """Return a meshgrid of resolution res."""

        res = res or self.res
        heatmap   = np.zeros((res+2*margin, res+2*margin))
        weightmap = np.zeros((res+2*margin, res+2*margin))

#        print list(self.self.dataset.iter_x())
        if self.extent is not None:
            xmin, xmax, ymin, ymax = self.extent
            bounds0 = xmin, xmax
            bounds1 = ymin, ymax
        else:
            bounds0 = min((y[0] for y in self.dataset.iter_x())), max((y[0] for y in self.dataset.iter_x()))
            bounds1 = min((y[1] for y in self.dataset.iter_x())), max((y[1] for y in self.dataset.iter_x()))
        
        for y, c in self.dataset.iter_xy():
            center = (margin - 1 + (y[0]-bounds0[0])/(max(1.0, bounds0[1]-bounds0[0]))*res, 
                      margin - 1 + (y[1]-bounds1[0])/(max(1.0, bounds1[1]-bounds1[0]))*res)
            neighborhood = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    neighborhood.append((int(center[0])+i, int(center[1])+j))
            for ng in neighborhood:
                d = toolbox.dist(ng, center)
                w = math.exp(-d*d/4)
                try:
                    heatmap[-ng[1]][ng[0]]   += w*c
                    weightmap[-ng[1]][ng[0]] += w
                except IndexError:
                    pass
        
        for i in range(res+2*margin):
            for j in range(res+2*margin):
                w = weightmap[i][j]
                if w > 0:
                    heatmap[i][j] /= 1+w
                if w == 0:
                    heatmap[i][j] = np.nan
        
        return heatmap, (bounds0, bounds1)    

    def plot(self, res = None):
        res = res or self.res
        heatmap, bounds = self._meshgrid(res = res)
        margin_x = margin*(bounds[0][1]-bounds[0][0])/res
        margin_y = margin*(bounds[1][1]-bounds[1][0])/res
        extent = [-margin_y + bounds[1][0],
                   margin_y + bounds[1][1],
                  -margin_x + bounds[0][0],
                   margin_x + bounds[0][1]]
        if self.subplot is not None:
            plt.subplot(self.subplot)
        cmap = cm.YlOrRd if not self.reversed else cm.YlOrRd_r
        im = plt.imshow(heatmap, interpolation = 'quadric', cmap = cmap,
                        extent = extent, aspect = 'auto')
        plt.colorbar(im)
        #plt.limits = 
        #plt.axes().set_aspect('equal')


class HotTestbed(object):

    def __init__(self, testbed, c_fun = c_fun_exp, res = 10):
        """Intialize the Heat Map
        @param testbed  the testbed from which to draw the map.
                        test should be generated from the testbed.
        @param c_fun    competence function
        """
        self.testbed = testbed
        self.c_fun   = c_fun
        self.res     = res

    def plot_fwd(self, res = None):
        res = res or self.res
        test_y, array_c = self._run_fwdtests()
        hm = HeatMap.from_xy_c(test_y, array_c)
        hm.plot(res = res)

    def plot_inv(self, res = None):
        res = res or self.res
        raise NotImplementedError
        test_y, array_c = self._run_fwdtests()
        hm = HeatMap.from_xy_c(test_y, array_c)
        hm.plot(res = res)

    def _run_fwdtests(self):
        errors = self.testbed.run_forward()
        map(self.c_fun, errors)
        return list(t[1] for t in self.testbed.testcases), errors

