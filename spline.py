# -*- coding: UTF-8 -*-
from __future__ import division

from numpy import array, dot, arange, linspace
from pylab import plot

class BSpline(object):
	
	def __init__(self, points, knots):
		self.points = points
		self.knots = knots
		self.length = len(knots) - len(points)
		self.degree = self.length + 1
	
	def find(self, t):
		# test this!
		return ((self.knots - t) > 0).argmax() - 1
	
	def plot_points(self):
		plot(self.points[:,0],self.points[:,1],'ro')
	
	plotres = 20
	
	def plot(self,a,b):
		ts = linspace(a,b,self.plotres)
		vals = array([self(t) for t in ts])
		plot(vals[:,0],vals[:,1])
	
	def __call__(self, t):
		left = self.find(t)
		pts = self.points[left-self.length:left+2]
		kns = self.knots[left - self.degree +1:left + self.degree + 1]
		for n in reversed(1+arange(self.degree)):
			diffs = kns[n:] - kns[:-n]
			lcoeff = (kns[n:] - t)/diffs
			rcoeff = (t - kns[:-n])/diffs
			pts = rcoeff.reshape(-1,1) * pts[1:] + lcoeff.reshape(-1,1) * pts[:-1]
			kns = kns[1:-1]
		return pts[0]

if __name__ == '__main__':
	ps = array([[1.,2], [2,3], [2,5], [1,6],[1,7]])
	knots = array([1.,2.,3.,4.,5.,6.,7.])
#	knots = array([3.,3.,3.,4.,4.,4.])
	s = BSpline(ps, knots)
	s(3.5)
	from pylab import clf
	clf()
	s.plot_points()
	s.plot(3.01,4.99)