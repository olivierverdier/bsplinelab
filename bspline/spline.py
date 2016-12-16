#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
from .geometry import Flat

def max_degree(nb_points):
    return nb_points - 1

def get_bezier_knots(degree):
    knots = np.zeros(2*degree)
    knots[degree:] = 1
    return knots

class Spline(object):
    def __init__(self, control_points, knots=None, geometry=Flat()):
        self.control_points = np.array(control_points)
        self.degree = max_degree(len(self.control_points))
        self.data_dim = np.ndim(self.control_points[0])

        if knots is None:
            knots = get_bezier_knots(self.degree)
        self.knots = knots

        self.interval = knots[self.degree-1], knots[self.degree]

        self.geometry = geometry

    @property
    def coeff_slice(self):
        """
        reshape the coefficients using data dimension and possible time shape
        for vectorial data, this amounts to the slice (:, np.newaxis,...)
        """
        return (slice(None),) + (np.newaxis,)*self.data_dim + (Ellipsis,)

    def iterate(self, t, pts, kns):
        n = len(kns)//2
        diffs = kns[n:] - kns[:-n] # (K,1)
        # trick to handle cases of equal knots:
        diffs[diffs==0.] = np.finfo(kns.dtype).eps
        rcoeff = (t - kns[:-n])/diffs # (K,T)
        pts = self.geometry.geodesic(pts[:-1], pts[1:], rcoeff[self.coeff_slice]) # (K, D, 1), (K, 1, T)
        kns = kns[1:-1]
        return pts, kns

    def __call__(self, t):
        t = np.array(t)
        kns = self.knots
        pts = self.control_points

        time_shape = (1,)*len(np.shape(t)) # time shape to add for broadcasting
        # we put the time on the last index
        kns = np.reshape(self.knots, self.knots.shape + time_shape) # (K, 1)
        pts = np.reshape(self.control_points, self.control_points.shape + time_shape) # (K, D, 1)


        degree = len(kns) - len(pts) + 1
        for i in range(degree):
            pts, kns = self.iterate(t, pts, kns)

        result = pts[0] # (D, T)
        # put time first by permuting the indices; in the vector case, this is a standard permutation
        permutation = len(np.shape(t))*(self.data_dim,) + tuple(range(self.data_dim))
        return result.transpose(permutation) # (T, D)
