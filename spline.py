#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

class Knots(object):
	"""
Knots class.

If:
	P = Nb points
	C = Nb curves
	K = Nb knots
	D = Degree

then the following relations are implemented in the code:

	C = K - 2D + 1
	D = K - P + 1

This also gives

	C + D = P
	K + 1 = P + D

For example, suppose that there are P=n+1 control points:

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
	def __init__(self, knots, degree=0):
		self.knots = np.array(knots, float)
		self.degree = degree

	def __repr__(self):
		return "<{} polynomials of degree {}>".format(self.nb_curves, self.degree)

	@property
	def nb_curves(self):
		return  len(self.knots) - 2*self.degree + 1

	def __getitem__(self, index):
		return self.knots[index]

	ktol = 1e-13

	def left_knot(self, t):
		"""
		Find out between which node a time t is.
		"""
		diff = self.knots[self.degree-1:-self.degree+1] - t
		isrightof = diff > self.ktol
		if np.all(isrightof):
			raise ValueError("Time too small")
		if np.all(~isrightof):
			raise ValueError("Time too big")
		left = np.argmax(isrightof) - 1 # argmax gives the right knot...
		return left + self.degree-1

	def abscissae(self):
		"""
		Return the Greville abscissae.
		"""
		if self.degree == 0:
			k = np.hstack([-1, self.knots])
			return k
		kernel = np.ones(self.degree)/self.degree
		res = np.convolve(kernel, self.knots, 'valid')
		return res

	def knot_range(self):
		"""
The range of knots from which to generate the points.
		"""
		if self.degree == 0:
			return []
		return range(self.degree - 1, self.degree - 1 + self.nb_curves)

	def intervals(self, knot_range=None):
		"""
		Compute the intervals from knot numbers `knot_range` till the next ones.
		"""
		if knot_range is None:
			knot_range = self.knot_range()
		for k in knot_range:
			width = self.knots[k+1]-self.knots[k]
			left, right = self.knots[k], self.knots[k+1]
			yield (k, left, right)

	def get_basis(self, k=None):
		if k is None:
			k = self.degree
		abscissae = self.abscissae()
		pts = np.zeros([len(abscissae), 2])
		pts[:,0] = self.abscissae()
		pts[k,1] = 1.
		return BSpline(self.knots, pts)


def geodesic(P1, P2, theta):
	"""
	The geodesic between two points.
	"""
	return (1-theta)*P1 + theta*P2

class BSpline(object):
	def __init__(self, knots, control_points):
		degree = len(knots) - len(control_points) + 1
		self.knots = Knots(knots, degree)
		self.control_points = np.array(control_points, float)

	def __call__(self, t, lknot=None):
		t = np.array(t)
		if lknot is None:
			lknot = self.knots.left_knot(t.flatten()[0])
		time_shape = (1,)*len(np.shape(t)) # time shape to add for broadcasting

		pts = self.control_points[lknot-self.knots.degree + 1:lknot+2]
		kns = self.knots[lknot - self.knots.degree + 1:lknot + self.knots.degree + 1]
		kns.shape = kns.shape + time_shape # (K, 1)
		if len(pts) != self.knots.degree + 1: # equivalent condition: len(kns) != 2*self.knots.degree
			raise ValueError("Wrong knot index.")

		# we put the time on the last index
		pts.shape = pts.shape + time_shape # (K, D, 1)

		for n in reversed(1+np.arange(self.knots.degree)):
			diffs = kns[n:] - kns[:-n] # (K,1)
			# trick to handle cases of equal knots:
			diffs[diffs==0.] = np.finfo(kns.dtype).eps
			rcoeff = (t - kns[:-n])/diffs # (K,T)
			pts = geodesic(pts[:-1], pts[1:], rcoeff[:,np.newaxis,...]) # (K, D, 1), (K, 1, T)
			kns = kns[1:-1]
		result = pts[0] # (D, T)
		return result.transpose() # (T, D)


	plotres = 200
	knot_style = {
			'marker':'o',
			'linestyle':'none',
			'markerfacecolor':'white',
			'markersize':5,
			'markeredgecolor':'black',
			}
	control_style={
 			'marker':'o',
			'linestyle':':',
			'color':'black',
			'markersize':10,
			'markerfacecolor':'white',
			'markeredgecolor':'red'
			}

	def plot_knots(self):
		ints = list(self.knots.intervals())
		pts = [self(l,k) for k,l,r in ints]
		pts.append(self(ints[-1][2], ints[-1][0])) # add last knot as well
		apts = np.array(pts)
		plt.plot(apts[:,0],apts[:,1], **self.knot_style)


	def plot_control_points(self):
		"""
		Plot the control points.
		"""
		plt.plot(self.control_points[:,0],self.control_points[:,1], **self.control_style)

	def plot(self, knot=None, with_knots=False, margin=0.):
		"""
		Plot the curve.
		"""
		self.plot_control_points()
		for k, left, right in self.knots.intervals(knot):
			ts = np.linspace(left, right, self.plotres)
			val = self(ts, lknot=k)
			plt.plot(val[:,0],val[:,1], label="{:1.0f} - {:1.0f}".format(self.knots[k], self.knots[k+1]), lw=2)
		if with_knots:
			self.plot_knots()



class Bezier(BSpline):
	"""
Special case of a BSpline. For n+1 points, the knot list is [0]*n+[1]*n.
	"""
	def __init__(self, control_points):
		degree = len(control_points) - 1
		knots = np.zeros(2*degree)
		knots[degree:] = 1
		super(Bezier,self).__init__(knots, control_points)

def get_basis_knots(x):
	"""
	Knots corresponding to the points in the array x.
	The corresponding basis is obtained by get_basis_knots(x).get_basis()
	"""
	x = np.array(x)
	degree = len(x) - 2
	knots = np.hstack([(degree-1)*[x[0]], x, (degree-1)*[x[-1]]])
	return Knots(knots, degree)


