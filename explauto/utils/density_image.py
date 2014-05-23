#!/usr/bin/env python

import numpy as np
import scipy.ndimage as ndi


def density_image(data_x, data_y, res_x, res_y, width_x, width_y, bounds, plot):
    bins_x = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / res_x)
    if bins_x[-1] < bounds[1] - 0.00001:
        bins_x = np.append(bins_x, bounds[1])
    bins_y = np.arange(bounds[2], bounds[3], (bounds[3] - bounds[2]) / res_y)
    if bins_y[-1] < bounds[3] - 0.00001:
        bins_y = np.append(bins_y, bounds[3])
    H, x_grid, y_grid = np.histogram2d(data_x, data_y, (bins_x, bins_y), normed=True)
    kde = ndi.gaussian_filter(H, [width_x, width_y], order=0)
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(kde.T[::-1, :], extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
        plt.axis([x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    else:
        return kde


