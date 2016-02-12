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
from mpl_toolkits.mplot3d import Axes3D
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

# Plot using bspline.plotting (currently projected 2D plot)
if True:
	fig1=plt.figure(1)
	splt.plot(b, with_control_points=False)
	fig1.suptitle('Spline on S^2, stereographically projected onto R^2')
	t=np.floor(N/2)

# Plot second derivative
if True:
	hh=1e-5
	ts = np.linspace(0.,N-1,(N-1)*100)[1:-1]
	ddb = np.zeros((ts.shape[0],3))
	for (tt,ddbt) in zip(ts,ddb):
	    ddbt[:] = (b(tt+hh)-2*b(tt)+b(tt-hh))/hh**2
	fig2 = plt.figure(2)
	plt.plot(ddb[:,0], ddb[:,1], linestyle='None', marker=',')
	fig2.suptitle('Second derivative of spline')

# Plot on the embedded sphere
if True:
	fig3=plt.figure(3)
	ax = fig3.add_subplot(111, projection='3d')
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	xx = 0.99*np.outer(np.cos(u), np.sin(v))
	yy = 0.99*np.outer(np.sin(u), np.sin(v))
	zz = 0.99*np.outer(np.ones(np.size(u)), np.cos(v))
	ax.plot_wireframe(xx, yy, zz, rstride=4, cstride=4, color='0.8', linewidth=1)
	ts = np.linspace(0.,N-1,1000)
	bvals = np.zeros((ts.shape[0],3))
	for (tt,bval) in zip(ts,bvals):
		bval[:] = b(tt)
	ax.plot(bvals[:,0],bvals[:,1],bvals[:,2],linewidth=2)
	fig3.suptitle('Spline on embedded sphere')


#fig3=plt.figure(3)
#plt.plot(ddb[(ts>t-0.1)*(ts<t+0.1),0], ddb[(ts>t-0.1)*(ts<t+0.1),1],linestyle='None', marker=',')
#fig3.suptitle('Zoom-in of second derivative around t=%4.2f'%(t))