# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:25:42 2016

@author: geirb
"""
from __future__ import division




import numpy as np


from bspline import geometry

from bspline.c2spline import implicitc2spline

import bspline.plotting as splt
import matplotlib.pyplot as plt




N=8
#interpolation_points = np.array([[1.,0,0], [0,0,1.], [0, np.sqrt(0.5), np.sqrt(0.5)], [0,1.,0]]) #spline interpolates these points
x = np.linspace(-1,1, N)
y = np.sin(5*np.pi*x)
interpolation_points =np.vstack([x,y,np.ones(x.shape)]).T
interpolation_points = interpolation_points/np.array([np.linalg.norm(interpolation_points, axis=1)]).T
#initial and end velocities:
init_vel = np.array([-1.0,0.0,-1.0])
end_vel = np.array([-1.0,0.0,1.0])
boundary_velocities=np.array([init_vel, end_vel])


b= implicitc2spline(interpolation_points, boundary_velocities, geometry=geometry.Sphere_geometry())
#h = np.power(10.0, range(-2,-6,-1))
#print((b(t+1.5*h)-3*b(t+0.5*h)+3*b(t-0.5*h)-b(t-1.5*h))/(h*h).reshape(h.shape +(1,))) 
#print("If C_2, these should approach zero.")


fig1=plt.figure(1)
splt.plot(b, with_control_points=False)
fig1.suptitle('Spline on S^2, stereographically projected onto R^2')

t=np.floor(N/2)


hh=1e-5
ts = np.linspace(0.,N-1, (N-1)*100)
ts= ts[1:-1]
ddb=np.array([]).reshape(0,3)
for tt in ts:
    temp=np.array((b(tt+hh)-2*b(tt)+b(tt-hh))/(hh*hh))
    temp=temp[np.newaxis, :]
    ddb = np.append(ddb, temp, axis=0) # There should be a better way of doing this.
fig2=plt.figure(2)
plt.plot(ddb[:,0],ddb[:, 1],linestyle='None', marker=',')
fig2.suptitle('Second derivative of spline')

#fig3=plt.figure(3)
#plt.plot(ddb[(ts>t-0.1)*(ts<t+0.1),0], ddb[(ts>t-0.1)*(ts<t+0.1),1],linestyle='None', marker=',')
#fig3.suptitle('Zoom-in of second derivative around t=%4.2f'%(t))