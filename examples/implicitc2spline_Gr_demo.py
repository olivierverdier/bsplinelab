# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:25:42 2016

@author: geirb
"""
from __future__ import division
import numpy as np
rng = np.random.default_rng()
from bspline import geometry
import matplotlib.pyplot as plt


alpha = 0.01
N=5
n=10
k=2
P1 = np.vstack((np.eye(k), np.zeros((n-k,k))))
interpolation_points= np.tile(P1, (N,1,1))

for i in range(1,N):
    P2 = P1+1.0*rng.standard_normal((n,k))
    interpolation_points[i] = np.linalg.qr(P2)[0]

from bspline.interpolation import cubic_spline, Riemann, Symmetric, Exponential


b = cubic_spline(InterpolationClass=Symmetric, interpolation_points=interpolation_points, geometry=geometry.Grassmannian())



if True:
    fig3=plt.figure(3)
    ax = fig3.add_subplot(111, projection='3d')
    tl = np.linspace(0.,N-1,1000)
    bvals =b(tl)
    bb = np.einsum('...ij,...kj', bvals,bvals)
    ax.plot(bb[:,0,0],bb[:,1,0],bb[:,2,0],linewidth=2)
    ax.plot(bb[:,0,1],bb[:,1,1],bb[:,2,1],linewidth=2)
    ax.plot(bb[:,0,2],bb[:,1,2],bb[:,2,2],linewidth=2 )
    ax.plot(np.array([0.]),np.array([0.]),np.array([0.]), marker = '.', markersize=20)


if True: # tests c2-continuity in interpolation point
    def bmat(t): return np.einsum('...ij,...kj',b(t), b(t))
    ttest = np.floor(0.5*N)
    H=np.power(2.0, range(-2,-20,-1))
    bmatdiv = (bmat(ttest+1.5*H)-3*bmat(ttest+0.5*H)+3*bmat(ttest-0.5*H)-bmat(ttest-1.5*H))/H[:, np.newaxis, np.newaxis]**2
    dd=np.linalg.norm(bmatdiv, axis=(1,2))
    fig6=plt.figure(6)
    plt.loglog(H, dd)
