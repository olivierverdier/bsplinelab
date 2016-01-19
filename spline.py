#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from knots import Knots

from geometry import flat_geodesic

class BSpline(object):
    def __init__(self, knots, control_points, geometry=flat_geodesic):
        degree = len(knots) - len(control_points) + 1
        self.knots = Knots(knots, degree)
        self.control_points = np.array(control_points, float)
        self.geometry = geometry


    def __call__(self, t, lknot=None):
        t = np.array(t)
        if lknot is None:
            lknot = self.knots.left_knot(t.flatten()[0])
        time_shape = (1,)*len(np.shape(t)) # time shape to add for broadcasting

        pts = self.control_points[lknot-self.knots.degree + 1:lknot+2]
        if len(pts) != self.knots.degree + 1: # equivalent condition: len(kns) != 2*self.knots.degree
            raise ValueError("Wrong knot index.")

        data_dim = np.ndim(pts[0]) # data dim to use for broadcasting

        kns = self.knots[lknot - self.knots.degree + 1:lknot + self.knots.degree + 1]
        kns.shape = kns.shape + time_shape # (K, 1)

        # we put the time on the last index
        pts.shape = pts.shape + time_shape # (K, D, 1)

        # reshape the coefficients using data dimension and possible time shape
        # for vectorial data, this amounts to the slice (:, np.newaxis,...)
        rcoeff_slice = (slice(None),) + (np.newaxis,)*data_dim + (Ellipsis,)

        for n in reversed(1+np.arange(self.knots.degree)):
            diffs = kns[n:] - kns[:-n] # (K,1)
            # trick to handle cases of equal knots:
            diffs[diffs==0.] = np.finfo(kns.dtype).eps
            rcoeff = (t - kns[:-n])/diffs # (K,T)
            pts = self.geometry(pts[:-1], pts[1:], rcoeff[rcoeff_slice]) # (K, D, 1), (K, 1, T)
            kns = kns[1:-1]
        result = pts[0] # (D, T)
        # put time first by permuting the indices; in the vector case, this is a standard permutation
        permutation = len(np.shape(t))*(data_dim,) + tuple(range(data_dim))
        return result.transpose(permutation) # (T, D)


    plotres = 200
    knot_style = {
            'marker':'o',
            'linestyle':'none',
            'markerfacecolor':'white',
            'markersize':5,
            'markeredgecolor':'black',
            }
    control_style={
            'marker':'o',
            'linestyle':':',
            'color':'black',
            'markersize':10,
            'markerfacecolor':'white',
            'markeredgecolor':'red'
            }

    def plot_knots(self):
        ints = list(self.knots.intervals())
        pts = [self(l,k) for k,l,r in ints]
        pts.append(self(ints[-1][2], ints[-1][0])) # add last knot as well
        apts = np.array(pts)
        plt.plot(apts[:,0],apts[:,1], **self.knot_style)


    def plot_control_points(self):
        """
        Plot the control points.
        """
        plt.plot(self.control_points[:,0],self.control_points[:,1], **self.control_style)

    def plot(self, knot=None, with_knots=False, margin=0.):
        """
        Plot the curve.
        """
        self.plot_control_points()
        for k, left, right in self.knots.intervals(knot):
            ts = np.linspace(left, right, self.plotres)
            val = self(ts, lknot=k)
            plt.plot(val[:,0],val[:,1], label="{:1.0f} - {:1.0f}".format(self.knots[k], self.knots[k+1]), lw=2)
        if with_knots:
            self.plot_knots()






