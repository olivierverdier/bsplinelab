#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
from .knots import Knots

from .geometry import flat_geodesic

class BSpline(object):
    def __init__(self, knots, control_points, geometry=flat_geodesic):
        degree = len(knots) - len(control_points) + 1
        self.knots = Knots(knots, degree)
        self.control_points = np.array(control_points) 
        self.geometry = geometry

    def data_dim(self):
        return np.ndim(self.control_points[0])

    def iterate(self, kns, pts, t):
        data_dim = self.data_dim() # data dim to use for broadcasting

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

    def __call__(self, t, lknot=None):
        t = np.array(t)
        if lknot is None:
            lknot = self.knots.left_knot(t.flatten()[0])
        time_shape = (1,)*len(np.shape(t)) # time shape to add for broadcasting

        pts = self.control_points[lknot-self.knots.degree + 1:lknot+2]
        if len(pts) != self.knots.degree + 1: # equivalent condition: len(kns) != 2*self.knots.degree
            raise ValueError("Wrong knot index.")

        kns = self.knots[lknot - self.knots.degree + 1:lknot + self.knots.degree + 1]
        kns.shape = kns.shape + time_shape # (K, 1)

        # we put the time on the last index
        pts.shape = pts.shape + time_shape # (K, D, 1)

        return self.iterate(kns, pts, t) # (D, T)



