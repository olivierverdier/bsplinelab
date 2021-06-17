#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
def plot_sphere(ax=None):
    if ax is None:
        ax = plt.gcf().add_subplot(111, projection='3d')
    ax.autoscale(tight=True)
    ax.set_axis_off()
    ax.set_aspect("auto")

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100).reshape(-1,1)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='k', alpha=.1)

    plt.tight_layout()
    return ax
