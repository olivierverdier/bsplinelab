# -*- coding: UTF-8 -*-
from __future__ import division

from numpy import array, dot, arange, linspace, nonzero, r_, isscalar, newaxis, squeeze
from pylab import plot, legend

from IPython.Debugger import Tracer
dh = Tracer()

class BSpline(object):
	
	def __init__(self, points, knots):
		self.points = points
		self.knots = knots
		self.length = len(knots) - len(points)
		self.degree = self.length + 1
	
	ktol = 1e-1
	
	def find(self, t):
		"""
		Unfinished function to find out between which node a time t is.
		"""
		raise DeprecationWarning
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
		"""
		Plot the control points.
		"""
		plot(self.points[:,0],self.points[:,1],'ro:')
	
## 	def plot_knots(self):
## 		kns = self.knots[self.length:-self.length]
## 		pts = array([self(kn,i) for i,kn in enumerate(kns[:-1])])
## 		plot(pts[:,0],pts[:,1],'sg')
	
	plotres = 200
	
	def compute(self, k_range=None):
		"""
		Compute the points from knot numbers k_range till the next ones.
		"""
		if k_range is None:
			k_range = range(self.length, len(self.knots)-self.length-1)
		res = []
		for k in k_range:
			left = self.knots[k]
			right = self.knots[k+1]
			times = linspace(left, right ,self.plotres * (right-left) + 1)
			res.append((times,k,self(times,k)))
		return res
	
	def plot(self, knot=None, with_knots=False):
		"""
		Plot the curve.
		"""
		self.plot_points()
		res = self.compute(knot)
		for t,k,val in res:
			plot(val[:,0],val[:,1], label="%1.f - %1.f" % (self.knots[k], self.knots[k+1]))
			if with_knots:
				plot(val[[0,-1],0], val[[0,-1],1], 'gs')
	
	def plot_basis(self):
		"""
		Plot the basis function for the given knot (or all of them)
		"""
		save_pts = self.points
		for k in k_range:
			self.points = zeros(len(self.knots))
			
		
	def __call__(self, t, lknot=None):
		if lknot is None:
			if isscalar(t):
				lknot = self.find(t)
			else:
				raise ValueError("A time array is only possible when the left knot is specified.")

		pts = self.points[lknot-self.length:lknot+2]
		kns = self.knots[lknot - self.degree +1:lknot + self.degree + 1]
		if len(kns) != 2*self.degree or len(pts) != self.length + 2:
			raise ValueError("Wrong knot index.")

		scalar_t = isscalar(t)
		if scalar_t:
			t = array([t])
		# we put the time on the first index; all other arrays must be reshaped accordingly
		t = t.reshape(-1,1,1)
		pts = pts[newaxis,...]
		
		for n in reversed(1+arange(self.degree)):
			diffs = kns[n:] - kns[:-n]
			lcoeff = (kns[n:] - t)/diffs
			rcoeff = (t - kns[:-n])/diffs
			pts = rcoeff.transpose(0,2,1) * pts[:,1:,:] + lcoeff.transpose(0,2,1) * pts[:,:-1,:]
			kns = kns[1:-1]
		result = pts[:,0,:]
		if scalar_t:
			result = squeeze(result) # test this
		return result
	
## class BSpline_basis(BSpline):
## 	def __init__(self, *args, **kwargs):
## 		super(BSpline_basis, self).__init__(self, *args, **kwargs)
## 		self.points = arange(len(self.points)).reshape(-1,1)
	
			

if __name__ == '__main__':
	ex1 = {
	'pts': array([[1.,2], [2,3], [2,5], [1,6]]),
	'knots': array([3.,3.,3.,4.,4.,4.])
	}

	ex2 = {
	'pts': array([[1.,2], [2,3], [2,5], [1,6], [1,9]]),
	'knots': array([1.,2.,3.,4.,5.,6.,7.])
	}
	
	# only C0
	ex3 = {
	'pts': array([[1.,2], [2,3], [2,5], [1,6], [1,9], [2,11], [2,9]]),
	'knots': array([1.,2.,3.,4.,4.,4.,5.,6.,6.])
	}
	
	# discontinuous
	ex4 = {
	'pts': array([[1.,2], [2,3], [2,5], [1,6], [1,9], [2,11], [2,9], [1.5,8]]),
	'knots': array([1.,2.,3.,4.,4.,4.,4.,5.,6.,6.])
	}
	
        deBoor=[[0.7,-0.4],[1.0,-0.4],[2.5,-1.2],[3.2,-.5],[-0.2,-.5],[.5,-1.2],[2.0,-.4],[2.3,-.4]]
        param=[1.,1.,1.,1.2,1.4,1.6,1.8,2.,2.,2.]

	ex_Claus = {}
	ex_Claus['pts'] = array(deBoor)
	ex_Claus['knots'] = array(param)

	ex = ex4
	
	s = BSpline(ex['pts'], ex['knots'])
	
	print s(array([3.2,3.5]), 2)
	from pylab import clf
	clf()
#	s.plot_points()
#	s.plot_knots()
	s.plot(with_knots=True)
	
## 	deBoor = array([[0.7,-0.4],
## 				[1.0,-0.4],
## 				[2.5,-1.2],
## 				[3.2,-.5],
## 				[-0.2,-.5],
## 				[.5,-1.2],
## 				[2.0,-.4],
## 				[2.3,-.4]])
## 	param = array([1.,1.,1.,1.2,1.4,1.6,1.8,2.,2.,2.])
## 	sc = BSpline(deBoor,param)
## 	sc.plot()
