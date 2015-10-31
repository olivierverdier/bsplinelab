#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

class BSpline(object):
	"""
BSpline class.

Suppose that there are n+1 control points:

======== ========= ====== =======
nb knots nb curves degree remarks
-------- --------- ------ -------
n        n+1       0      n+1 points
n+1      n         1      n segments
n+2      n-1       2
...      ...       ...
2n       1         n      if first n and last n knots are equal: Bézier case
-------- --------- ------ -------
	"""
	def __init__(self, control_points, knots):
		self.control_points = np.array(control_points, float)
		self.knots = np.array(knots, float)
		self.compute_info()

	def compute_info(self):
		"""
Compute information about nb of curves and degree.
		"""
		nb_control_points = len(self.control_points)
		self.length = len(self.knots) - nb_control_points
		self.degree = self.length + 1
		self.nb_curves = nb_control_points - self.degree

	ktol = 1e-13

	def left_knot(self, t):
		"""
		Find out between which node a time t is.
		"""
		diff = self.knots[self.length:-self.length] - t
		isrightof = diff > self.ktol
		if np.all(isrightof):
			raise ValueError("Time too small")
		if np.all(~isrightof):
			raise ValueError("Time too big")
		left = np.argmax(isrightof) - 1 # argmax gives the right knot...
		return left + self.length

	def plot_control_points(self):
		"""
		Plot the control points.
		"""
		plt.plot(self.control_points[:,0],self.control_points[:,1], marker='o', ls=':', color='black', markersize=10, mfc='white', mec='red')

## 	def plot_knots(self):
## 		kns = self.knots[self.length:-self.length]
## 		pts = np.array([self(kn,i) for i,kn in enumerate(kns[:-1])])
## 		plot(pts[:,0],pts[:,1],'sg')

	plotres = 200

	def knot_range(self):
		"""
The range of knots from which to generate the points.
		"""
		if self.length < 0:
			return []
		return range(self.length, self.length + self.nb_curves)

	def generate_points(self, knot_range=None, margin=0.):
		"""
		Compute the points from knot numbers `knot_range` till the next ones.
		"""
		if knot_range is None:
			knot_range = self.knot_range()
		for k in knot_range:
			width = self.knots[k+1]-self.knots[k]
			extra = margin*width
			left, right = self.knots[k]-margin, self.knots[k+1]+margin
			times = np.linspace(left, right, self.plotres)
			yield (times,k,self(times,k,))


	def plot(self, knot=None, with_knots=False, margin=0.):
		"""
		Plot the curve.
		"""
		self.plot_control_points()
		for t,k,val in self.generate_points(knot, margin):
			plt.plot(val[:,0],val[:,1], label="{:1.0f} - {:1.0f}".format(self.knots[k], self.knots[k+1]), lw=2)
			if with_knots:
				plt.plot(val[[0,-1],0], val[[0,-1],1], marker='o', ls='none', markerfacecolor='white', markersize=5, markeredgecolor='black')


	def __call__(self, t, lknot=None):
		if lknot is None:
			if np.isscalar(t):
				lknot = self.left_knot(t)
			else:
				raise ValueError("A time array is only possible when the left knot is specified.")

		pts = self.control_points[lknot-self.length:lknot+2]
		kns = self.knots[lknot - self.degree +1:lknot + self.degree + 1]
		if len(kns) != 2*self.degree or len(pts) != self.length + 2:
			raise ValueError("Wrong knot index.")

		scalar_t = np.isscalar(t)
		if scalar_t:
			t = np.array([t])
		# we put the time on the first index; all other arrays must be reshaped accordingly
		t = t.reshape(-1,1,1)
		pts = pts[np.newaxis,...]

		for n in reversed(1+np.arange(self.degree)):
			diffs = kns[n:] - kns[:-n]
			# trick to handle cases of equal knots:
			diffs[diffs==0.] = np.finfo(kns.dtype).eps
			rcoeff = (t - kns[:-n])/diffs
			pts = geodesic(pts[:,:-1,:], pts[:,1:,:], rcoeff.transpose(0,2,1))
			kns = kns[1:-1]
		result = pts[:,0,:]
		if scalar_t:
			result = np.squeeze(result) # test this
		return result

def geodesic(P1, P2, theta):
	"""
	The geodesic between two points.
	"""
	return (1-theta)*P1 + theta*P2

class Bezier(BSpline):
	"""
Special case of a BSpline. For n+1 points, the knot list is [0]*n+[1]*n.
	"""
	def __init__(self, control_points):
		nb_control_points = len(control_points)
		self.rknot = nb_control_points-1
		knots = np.zeros(2*self.rknot)
		knots[self.rknot:] = 1
		super(Bezier,self).__init__(control_points,knots)

	def __call__(self, t, k=None):
		return super(Bezier,self).__call__(t, lknot=self.rknot-1)

def get_basis(n):
	nb_pts = 2*n+1
	knots = np.arange(nb_pts+n-1)
	control_points = np.vstack([np.arange(nb_pts),np.zeros(nb_pts)]).T
	control_points[n,1] = 1.

	spline = BSpline(control_points, knots)
	return spline

from scipy.linalg import toeplitz

def greville(knots):
	n = len(knots)//2
	col = np.zeros(len(knots) - (n-1))
	col[0] = 1
	row = np.zeros_like(knots)
	row[:n] = 1
	mat = toeplitz(col, row)/n
	return mat

def plot_basis(x, h=1.):
	n = len(x)
	degree = n-2
	regularity = degree - 1
	if regularity % 2: # even degree
		y = (x[:-1] + x[1:])/2
	else:
		y = x
	extra_points = (np.arange(degree//2)+1.)*h
	points = np.hstack([x[0] - extra_points, y, x[-1]+extra_points])
	return len(points)




def noplot_basis(x, h=1.):
	degree = len(x) - 2
	regularity = degree - 1
	mat = np.zeros([len(x), degree*(len(x)-1) +1])
	elem = np.vstack([np.arange(1,degree+1)[::-1], np.arange(degree)])
	for i in range(len(x)-1):
		mat[i:i+2,degree*i:degree*(i+1)] = elem
	mat[-1,-1] = degree
	extra_points = np.dot(x,mat)/degree
	points = np.vstack([extra_points,np.zeros_like(extra_points)])
	points[1,len(extra_points)//2] = 1.
	knots = np.array([x]*degree).T.reshape(-1)
	spline = BSpline(points.T, knots)
	spline.plot()
	return spline


