#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy.testing as npt
import unittest

from spline import *

class TestBezier(unittest.TestCase):
	def setUp(self):
		controls = [[1.,1],[0,-1],[-1,1]]
		self.b = Bezier(controls)

	def test_quad(self):
		u"""
	Check that Bézier with three points generates the parabola y=x**2.
		"""
		b = self.b
		self.assertEqual(b.nb_curves,1)
		ts = np.linspace(0.,1., 200)
		all_pts = b(ts)
		npt.assert_array_almost_equal(all_pts[:,0]**2, all_pts[:,1])
		npt.assert_allclose(b(.5), 0.)

	def test_generate(self):
		pt_list = list(pts for (t,k,pts) in self.b.generate_points())
		self.assertEqual(len(pt_list), self.b.nb_curves)

	def test_left_knot(self):
		self.assertEqual(self.b.left_knot(.2), 1)
		self.assertEqual(self.b.left_knot(.8), 1)


class Test_DoubleQuad(unittest.TestCase):
	def setUp(self):
		controls = [[-1,1],[0,-1],[2.,3],[3,1]]
		knots = [0,0,.5,1,1]
		self.spline = BSpline(controls, knots)

	def test_info(self):
		self.assertEqual(self.spline.degree, 2)
		self.assertEqual(self.spline.nb_curves,2)
		self.assertEqual(len(self.spline.knot_range()), self.spline.nb_curves)

	def test_generate(self):
		gen_pts = list(self.spline.generate_points())
		self.assertEqual(len(gen_pts), self.spline.nb_curves)
		a0,a1 = np.array(gen_pts[0][2]), np.array(gen_pts[1][2])
		npt.assert_array_almost_equal(a0[:,0]**2, a0[:,1])
		npt.assert_array_almost_equal(-(a1[:,0]-2)**2, a1[:,1]-2)
		npt.assert_allclose(self.spline(gen_pts[0][0]), gen_pts[0][2])

class Test_BSpline(unittest.TestCase):
	def setUp(self):
		ex2 = {
		'control_points': np.array([[1.,2], [2,3], [2,5], [1,6], [1,9]]),
		'knots': np.array([1.,2.,3.,4.,5.,6.,7.])
		}
		self.b = BSpline(**ex2)

	def test_left_knot(self):
		self.assertEqual(self.b.left_knot(3.8), 2)
		self.assertEqual(self.b.left_knot(3.2), 2)
		self.assertEqual(self.b.left_knot(4.8), 3)
		self.assertEqual(self.b.left_knot(4.0), 3)
		self.assertEqual(self.b.left_knot(4.0-1e-14), 3)
		with self.assertRaises(ValueError):
			self.b.left_knot(2.5)
		with self.assertRaises(ValueError):
			self.b.left_knot(5.5)

class Test_BSpline3(unittest.TestCase):
	def setUp(self):
		ex2 = {
		'control_points': np.array([[1.,2,0], [2,3,1], [2,5,3], [1,6,3], [1,9,2]]),
		'knots': np.array([1.,2.,3.,4.,5.,6.,7.])
		}
		self.b = BSpline(**ex2)

	def test_call(self):
		self.b(3.5)

	@unittest.expectedFailure
	def test_scalar_shape(self):
		self.assertEqual(np.shape(self.b(3.5)), (1,))
