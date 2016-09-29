#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

from .geometry import Geometry
from .spline import Spline, max_degree, get_bezier_knots

class BSpline(object):
    def __init__(self, control_points, knots=None, geometry=Geometry()):
        if knots is None:
            self.degree = max_degree(len(control_points))
            self.knots = get_bezier_knots(self.degree)
        else:
            self.knots = knots
            self.degree = len(knots) - len(control_points) + 1
        self.control_points = np.array(control_points)
        self.geometry = geometry
        self._splines = [Spline(control_points=pts,
                                knots=kns,
                                geometry=self.geometry)
                         for pts, kns in get_splines_data(self.knots, self.control_points)]

    def __repr__(self):
        return "<{} splines of degree {}>".format(len(self), self.degree)

    def __len__(self):
        return len(self._splines)

    def __call__(self, t):
        t=np.asanyarray(t)
        l=[]
        a0 =self._splines[0].interval[0]    
        bn = self._splines[-1].interval[1]
        if (t<a0).any() or (t>bn).any():
            raise ValueError("Outside interval")
        for s in self:
            a,b = s.interval
            ts = t[(t>=a)*(t<b)]
            l.append(s(ts))
        ts=t[t==b]
        l.append(s(ts))
        return np.squeeze(np.concatenate(l))

    def __iter__(self):
        for spline in self._splines:
            yield spline


def get_single_bspline(spline):
    return BSpline(knots=spline.knots, control_points=spline.control_points)



def get_splines_data(knots, points):
    degree = len(knots) - len(points) + 1
    nb_curves = len(points) - degree
    for k in range(nb_curves):
        kns = knots[k:k+2*degree]
        pts = points[k:k+degree+1]
        yield pts, kns
