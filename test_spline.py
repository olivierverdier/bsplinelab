#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy.testing as npt
import nose.tools as nt

from spline import *

def test_quad():
	ur"""
Check that Bézier with three points generates the parabola y=x**2.
	"""
	controls = [[1.,1],[0,-1],[-1,1]]
	b = Bezier(controls)
	pt_list = list(pts for (t,k,pts) in b.generate_points())
	nt.assert_equal(len(pt_list),1)
	nt.assert_equal(b.nb_curves,1)
	all_pts = np.vstack(pt_list)
	npt.assert_array_almost_equal(all_pts[:,0]**2, all_pts[:,1])

class Test_DoubleQuad(object):
	def setUp(self):
		controls = [[-1,1],[0,-1],[2.,3],[3,1]]
		knots = [0,0,.5,1,1]
		self.spline = BSpline(controls, knots)

	def test_info(self):
		nt.assert_equal(self.spline.degree, 2)
		nt.assert_equal(self.spline.nb_curves,2)
		nt.assert_equal(len(self.spline.knot_range()), self.spline.nb_curves)

	def test_points(self):
		pt_list = [pts for t,k,pts in self.spline.generate_points()]
		nt.assert_equal(len(pt_list), self.spline.nb_curves)
		a0,a1 = np.array(pt_list[0]), np.array(pt_list[1])
		npt.assert_array_almost_equal(a0[:,0]**2, a0[:,1])
		npt.assert_array_almost_equal(-(a1[:,0]-2)**2, a1[:,1]-2)


