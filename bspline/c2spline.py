
#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

from .geometry import Geometry
from .spline import BSpline

import warnings

def c2spline(interpolation_points, initial_control_points, geometry = Geometry()):
    # initialize the array for control points of the final spline.
    S=list(interpolation_points.shape)
    S[0]=3*S[0]-2
    control_points=np.zeros(S) 
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

def implicitc2spline(interpolation_points, boundary_velocities, geometry=Geometry(), Maxiter=500):
    N = interpolation_points.shape[0]
    S=list(interpolation_points.shape)
    S[0]=3*S[0]-2
    control_points=np.zeros(S) 
    control_points[::3]=interpolation_points
    
    velocities=np.zeros(interpolation_points.shape)
    velocities[[0,-1]]=boundary_velocities/3.0
    for i in range(0,N-1):
        control_points[3*i+1]=geometry.exp(interpolation_points[i], velocities[i])
    for i in range(1,N):
        control_points[3*i-1]=geometry.exp(interpolation_points[i], -velocities[i])
    err= np.inf
    tol = 16*N*np.finfo(float).eps
    Niter = 0
    for Niter in range(Maxiter):
        old_velocities=np.copy(velocities)
        err=0
        for i in range(1,N-1):
            wplus = geometry.log(control_points[3*i+1], control_points[3*i+2])
            wminus = geometry.log(control_points[3*i-1], control_points[3*i-2])
            fi=geometry.dexpinv(interpolation_points[i], old_velocities[i], wplus)\
                -geometry.dexpinv(interpolation_points[i], -old_velocities[i], wminus)-2*old_velocities[i]
            err = err+np.linalg.norm(fi)
            velocities[i]=old_velocities[i]+0.25*(fi)
        for i in range(1,N-1):
            control_points[3*i+1]=geometry.exp(interpolation_points[i], velocities[i])
            control_points[3*i-1]=geometry.exp(interpolation_points[i], -velocities[i])
        if err < tol:
            break
    else:
        raise Exception("No convergence in {} steps".format(Niter))
    print("#iterations: "+str(Niter))
    print("Error: "+str(err))
    ex = {
    'control_points': control_points,
    'knots' : np.array(range(interpolation_points.shape[0]), dtype=float).repeat(3)
    }
    return BSpline(geometry=geometry, **ex)
     
