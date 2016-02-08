
#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

from .geometry import Geometry
from .spline import BSpline

def c2spline(interpolation_points, initial_control_points, geometry = Geometry()):
    # initialize the array for control points of the final spline.
    control_points=np.zeros([interpolation_points.shape[0]*3-2, interpolation_points.shape[1]]) 
    # Set up for main loop
    P = interpolation_points[0]
    prev_control_points = initial_control_points[0:2]
    new_control_points=np.zeros(prev_control_points.shape)
    # Copy initial data to control_points
    control_points[0] = P
    control_points[1:3]=prev_control_points

    k=3
    
    for P in interpolation_points[1:-1]:
        # Calculate the new control points
        new_control_points[0]=geometry.involution(P, prev_control_points[1])
        w=geometry.log(prev_control_points[1], prev_control_points[0])
        u = -geometry.involution_derivative(P, prev_control_points[1], w)-2*geometry.log(new_control_points[0], P)
        new_control_points[1]=geometry.exp(new_control_points[0], u)
        
        #copy data to control_points
        control_points[k] = P
        control_points[k+1:k+3]=new_control_points
        k=k+3
        prev_control_points=np.copy(new_control_points)
    #insert final interpolation point at end
    control_points[k]=interpolation_points[-1]
    
    ex = {
    'control_points': control_points,
    'knots' : np.array(range(interpolation_points.shape[0]), dtype=float).repeat(3)
    }
    return BSpline(geometry=geometry, **ex)
