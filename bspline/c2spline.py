
#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

from .geometry import Geometry
from .spline import BSpline

def c2spline(interpolation_points, initial_control_points, geometry = Geometry()):
    
    P = interpolation_points[0]
    prev_control_points = initial_control_points[0:2]
    k=0
    control_points=np.zeros([interpolation_points.shape[0]*3-2, interpolation_points.shape[1]])
    control_points[k] = P
    control_points[k+1:k+3]=prev_control_points
    w=geometry.log(prev_control_points[1], prev_control_points[0])
    k=k+3
    new_control_points=np.zeros(prev_control_points.shape)
    for P in interpolation_points[1:-1]:
        new_control_points[0]=geometry.involution(P, prev_control_points[1])
        u = -geometry.involution_derivative(P, prev_control_points[1], w)-2*geometry.log(new_control_points[0], P)
        new_control_points[1]=geometry.exp(new_control_points[0], u)
        prev_control_points=new_control_points
        control_points[k] = P
        control_points[k+1:k+3]=prev_control_points
        w=geometry.log(prev_control_points[1], prev_control_points[0])
        k=k+3
    control_points[-1]=interpolation_points[-1]
    
    ex = {
    'control_points': control_points,
    'knots' : np.array(range(interpolation_points.shape[0])).repeat(3)
    }
    return BSpline(**ex, geometry=geometry)
