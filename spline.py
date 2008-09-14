# -*- coding: UTF-8 -*-
from __future__ import division

from numpy import array, dot, arange, linspace, nonzero
from pylab import plot

class BSpline(object):
	
	def __init__(self, points, knots):
		self.points = points
		self.knots = knots
		self.length = len(knots) - len(points)
		self.degree = self.length + 1
	
	ktol = 1e-1
	
	def find(self, t):
		# test this!
		dist = self.knots - t
		# test if it's right on a knot
		on_knot = abs(dist)
		nearest = on_knot.argmin()
		# if the nearest is on the right, return the previous one:
		knot = self.knots[nearest]
		if t < knot:
			nearest -= 1
# 		if nearest not in range(self.length,len(self.knots) + )
		return nearest
## 		# otherwise take the left node
## 		candidate = (dist > 0).argmax() - 1
## 		return candidate
	
	def plot_points(self):
		plot(self.points[:,0],self.points[:,1],'ro')
	
	def plot_knots(self):
		kns = self.knots[self.length:-self.length]
		pts = array([self(kn) for kn in kns[:-1]])
		plot(pts[:,0],pts[:,1],'sg')
	
	plotres = 200
	
	def plot(self,a=None,b=None):
		self.plot_points()
		self.plot_knots()
		if a is None or b is None:
			a = self.knots[self.length]+self.ktol
			b = self.knots[-self.length]-self.ktol
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
	ps = array([[1.,2], [2,3], [2,5], [1,6],[1,9]])
	knots = array([1.,2.,3.,4.,5.,5.,5.])
#	knots = array([3.,3.,3.,4.,4.,4.])
	s = BSpline(ps, knots)
	s(3.5)
	from pylab import clf
	clf()
#	s.plot_points()
#	s.plot_knots()
#	s.plot()#(3.01,4.99)
	
	deBoor = array([[0.7,-0.4],
				[1.0,-0.4],
				[2.5,-1.2],
				[3.2,-.5],
				[-0.2,-.5],
				[.5,-1.2],
				[2.0,-.4],
				[2.3,-.4]])
	param = array([1.,1.,1.,1.2,1.4,1.6,1.8,2.,2.,2.])
	sc = BSpline(deBoor,param)
	sc.plot()
