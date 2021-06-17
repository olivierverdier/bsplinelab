# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:30:15 2016

@author: geirb
"""

flat = False


import numpy as np


from bspline import BSpline

from bspline import geometry
import bspline.plotting as splt
import matplotlib.pyplot as plt
from bspline.plotting import control_style


def Invol(p,q): #involution operator on sphere Invol(p,q) mirrors q through p.
    if flat:
        return 2*p-q
    else:
        return 2*np.inner(p.conj(), q).real*p-q

interpolation_points = np.array([[1.,0,0], [0,0,1.], [0,1.,0]]) #spline interpolates these points
# control points for first cubic spline
x01 = np.array([0.2,-0.5,1])
x01=x01/np.linalg.norm(x01)
x02 = np.array([0, -0.3, 1])
x02 = x02/np.linalg.norm(x02)
control_points1 = np.array([x01,x02])

auxiliary_point = Invol(control_points1[1,:], control_points1[0,:]) # auxiliary point
x11 =Invol(interpolation_points[1,:], control_points1[1,:]) 
x12 =Invol(x11, auxiliary_point)
control_points2=np.array([x11,x12]) # control points for second cubic spline
ex={
'control_points': np.vstack((interpolation_points[0,:],  control_points1, interpolation_points[1,:], control_points2, interpolation_points[2,:])),
'knots': np.array([0.,0.,0.,1.,1.,1.,2.,2.,2.])
}

if flat:
    b = BSpline(**ex)
else:
    b = BSpline(geometry=geometry.Sphere(), **ex)

plt.figure(1)
splt.plot(b)
plt.plot([x02[0],auxiliary_point[0],x11[0]], [x02[1],auxiliary_point[1],x11[1]], **control_style)
t=1.
h = np.power(10.0, range(-2,-7,-1))
print((b(t+1.5*h)-3*b(t+0.5*h)+3*b(t-0.5*h)-b(t-1.5*h))/(h*h).reshape(h.shape +(1,))) 
print("If C_2, these should approach zero.")

hh=1e-5
ts = np.linspace(0.,2., 2*2000)
ts= ts[1:-1]
ddb=np.array([]).reshape(0,3)
for tt in ts:
    temp=np.array((b(tt+hh)-2*b(tt)+b(tt-hh))/(hh*hh))
    temp=temp[np.newaxis, :]
    ddb = np.append(ddb, temp, axis=0) # There should be a better way of doing this.
plt.figure(2)
plt.plot(ddb[:,0],ddb[:, 1],linestyle='None', marker=',')
plt.figure(3)
plt.plot(ddb[(ts>0.9)*(ts<1.1),0], ddb[(ts>0.9)*(ts<1.1),1],linestyle='None', marker=',')

