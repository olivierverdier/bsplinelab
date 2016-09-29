# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:52:48 2016

@author: geirb
"""

from __future__ import division
import numpy as np
from bspline import geometry
from bspline.c2spline import implicitc2spline
import bspline.plotting as splt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def cp1tobloch(u): # mapping from CP_1 to Bloch sphere
    theta = 2*np.arccos(np.clip(np.abs(u[0]),-1,1))
    phi = np.angle(u[1])-np.angle(u[0])
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def bbloch(b, tt):
    return cp1tobloch(b(tt))

d=2
N=4
interpolation_points =np.random.randn(N, d)+ 1j*np.random.randn(N,d)
interpolation_points = interpolation_points/np.array([np.linalg.norm(interpolation_points, axis=1)]).T
#interpolation_points=np.array([[1,0], [np.sqrt(0.5),np.sqrt(0.5)*1j], [1,0]])
#initial and end velocities:
init_vel = np.zeros(d, dtype='complex128')
end_vel = np.zeros(d, dtype='complex128')
boundary_velocities=np.array([init_vel, end_vel])


b= implicitc2spline(interpolation_points, geometry=geometry.CP_geometry(), Maxiter=10000)
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
    ddb = np.zeros((ts.shape[0],d), dtype=interpolation_points.dtype)
    pm1=b(ts-hh)
    p0=b(ts)
    p1=b(ts+hh)
    am = np.einsum('ij,ij->i',pm1.conj(), p0)
    ap = np.einsum('ij,ij->i', p1.conj(), p0)
    am=am[..., np.newaxis]
    ap=ap[...,np.newaxis]
    ddb = (am/np.abs(am)*pm1+ap/np.abs(ap)*p1-2*p0) /hh**2
    fig2 = plt.figure(2)
    plt.plot(ts, np.angle(ddb), linestyle='None', marker=',')
    fig2.suptitle('Second derivative of spline')
    



# Plot on the embedded sphere
if d==2:
    fig3=plt.figure(3)
    ax = fig3.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xx = 0.99*np.outer(np.cos(u), np.sin(v))
    yy = 0.99*np.outer(np.sin(u), np.sin(v))
    zz = 0.99*np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xx, yy, zz, rstride=4, cstride=4, color='0.8', linewidth=1)
    ts = np.linspace(0.,N-1,1000)
    bvals = cp1tobloch(b(ts).T).T
    ax.plot(bvals[:,0],bvals[:,1],bvals[:,2],linewidth=2)
    fig3.suptitle('Spline on embedded sphere')


if True:
    
    fig4 = plt.figure(4)
    ts = np.linspace(0.,N-1,(N-1)*100)[1:-1]
    bv=b(ts)
    logbv=np.log(bv)
    plt.plot(ts, logbv.real)
    fig5=plt.figure(5)
    plt.plot(ts, logbv.imag)


if True: # tests c2-continuity in interpolation point
    def bmat(t): return 1j*np.einsum('ij,ik->ijk',b(t), b(t).conj())
    ttest = np.floor(0.5*N)
    H=np.power(2.0, range(-2,-20,-1))
    bmatdiv = (bmat(ttest+1.5*H)-3*bmat(ttest+0.5*H)+3*bmat(ttest-0.5*H)-bmat(ttest-1.5*H))/H[:, np.newaxis, np.newaxis]**2
    dd=np.linalg.norm(bmatdiv, axis=(1,2))    
    fig6=plt.figure(6)
    plt.loglog(H, dd)

if True: # tests zero acceleration at start-point
    def bmat(t): return 1j*np.outer(b(t), b(t).conj())
    def ddmat(t,h): return (bmat(t+h)-2*bmat(t)+bmat(t-h))/h[..., np.newaxis, np.newaxis]**2
    ttest = 0
    ide = np.eye(d)
    H2=np.power(2.0, range(-2,-20,-1))
    dd2 = np.zeros_like(H2)
    dd2=dd2[:, np.newaxis]
    for (hh,dl) in zip(H2,dd2):
        dl[:] = np.linalg.norm(((bmat(hh)-1j*ide).dot(ddmat(hh,hh)).dot(bmat(hh))+bmat(hh).dot(ddmat(hh,hh)).dot(bmat(hh)-1j*ide)))
    dd2=dd2.flatten()
    fig7=plt.figure(7)
    plt.loglog(H2, dd2)
        
    
