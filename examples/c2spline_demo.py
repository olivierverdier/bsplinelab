# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:25:42 2016

@author: geirb
"""
from __future__ import division




import numpy as np


from bspline import geometry

import bspline.plotting as splt
import matplotlib.pyplot as plt

from bspline.interpolation import cubic_spline, Riemann, Symmetric, Exponential




interpolation_points = np.array([[1.,0,0], [0,0,1.], [0,1.,0], [1.,0,0]]) #spline interpolates these points

# control points for first cubic spline:
x01 = np.array([0.2,-0.5,1])
x01=x01/np.linalg.norm(x01)
x02 = np.array([0, -0.3, 1])
x02 = x02/np.linalg.norm(x02)
initial_control_points= np.array([x01,x02])
#b = c2spline(interpolation_points, initial_control_points, geometry=geometry.Sphere_geometry())
b = cubic_spline(Exponential, interpolation_points, initial_control_points, geometry=geometry.Sphere())

t=2.
h = np.power(10.0, range(-2,-6,-1))
print((b(t+1.5*h)-3*b(t+0.5*h)+3*b(t-0.5*h)-b(t-1.5*h))/(h*h).reshape(h.shape +(1,))) 
print("If C_2, these should approach zero.")


fig1=plt.figure(1)
splt.plot(b)
fig1.suptitle('Spline on S^2, stereographically projected onto R^2')

import tqdm

hh=1e-5
ts = np.linspace(0.,3., 3*2000)
ts= ts[1:-1]
ddb=np.array([]).reshape(0,3)
for tt in tqdm.tqdm(ts):
    temp=np.array((b(tt+hh)-2*b(tt)+b(tt-hh))/(hh*hh))
    temp=temp[np.newaxis, :]
    ddb = np.append(ddb, temp, axis=0) # There should be a better way of doing this.
fig2=plt.figure(2)
plt.plot(ddb[:,0],ddb[:, 1],linestyle='None', marker=',')
fig2.suptitle('Second derivative of spline')

fig3=plt.figure(3)
plt.plot(ddb[(ts>t-0.1)*(ts<t+0.1),0], ddb[(ts>t-0.1)*(ts<t+0.1),1],linestyle='None', marker=',')
fig3.suptitle('Zoom-in of second derivative around t=%4.2f'%(t))
