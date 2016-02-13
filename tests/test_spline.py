#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy.testing as npt
import unittest

import numpy as np

from bspline import BSpline
from bspline.bspline import Spline, get_single_bspline
from bspline import plotting
from bspline import geometry

def sphere_geodesic_unstable(P1,P2,theta):
    """
    Geodesic on the 2n+1-sphere, undefined when P1 == P2
    """
    angle = np.arccos(np.einsum('ij...,ij...->i...',P1.conj(), P2).real)
    angle = angle[:, np.newaxis,...]
    return (np.sin((1-theta)*angle)*P1 + np.sin(theta*angle)*P2)/np.sin(angle)



def get_basis(n):
    nb_pts = 2*n+1
    knots = np.arange(3*n) - (3*n-1)/2
    control_points = np.vstack([np.arange(nb_pts),np.zeros(nb_pts)]).T
    control_points[n,1] = 1.

    spline = BSpline(knots, control_points)
    return spline

class TestGeodesic(unittest.TestCase):
    @unittest.skip("This will not work with current implementation of geodesic")
    def test_single_geodesic(self):
        G = geometry.Geometry()
        P1 = np.array([0.,0])
        P2 = np.array([1.,1])
        theta = np.linspace(0,1)
        return G.geodesic(P1,P2,theta)


class TestBezier(unittest.TestCase):
    def setUp(self):
        self.controls = np.array([[1.,1],[0,-1],[-1,1]])
        self.b = Spline(self.controls)

    def test_quad(self):
        u"""
    Check that Bézier with three points generates the parabola y=x**2.
        """
        b = self.b
        # self.assertEqual(b.knots.nb_curves,1)
        ts = np.linspace(0.,1., 200)
        all_pts = b(ts)
        npt.assert_array_almost_equal(all_pts[:,0]**2, all_pts[:,1])
        npt.assert_allclose(b(.5), 0.)

    def test_plot(self):
        b = get_single_bspline(self.b)
        plotting.plot(b)

    def test_create_bezier(self):
        b_ = BSpline(self.controls)


class TestDoubleQuadSpline(unittest.TestCase):
    def setUp(self):
        controls = np.array([[-1,1],[0,-1],[2.,3],[3,1]])
        self.knots = np.array([0,0,.5,1,1])
        self.spline = BSpline(knots=self.knots, control_points=controls)

    def test_generate(self):
        a0,a1 = [s(np.linspace(s.interval[0],s.interval[1],200)) for s in self.spline]
        npt.assert_array_almost_equal(a0[:,0]**2, a0[:,1])
        npt.assert_array_almost_equal(-(a1[:,0]-2)**2, a1[:,1]-2)

    def test_outside_interval(self):
        with self.assertRaises(ValueError):
            self.spline(10.)



class Test_BSpline(unittest.TestCase):
    def setUp(self):
        ex2 = {
        'control_points': np.array([[1.,2], [2,3], [2,5], [1,6], [1,9]]),
        'knots': np.array([1.,2.,3.,4.,5.,6.,7.])
        }
        self.b = BSpline(**ex2)

    @unittest.skip("Obselete test")
    def test_wrong_knot(self):
        with self.assertRaises(ValueError):
            self.b(3.5, lknot=1)
        with self.assertRaises(ValueError):
            self.b(3.5, lknot=4)

    @unittest.skip("fix later")
    def test_vectorize(self):
        control_points = np.array([0.,1.]*2)
        knots = np.arange(5)
        s = BSpline(knots, control_points)
        ts = np.array([1.5,2.5])
        ss_ = np.array([s(tt) for tt in ts])
        ss = s(ts)
        npt.assert_allclose(ss[-1], ss_[-1])



    def test_plot(self):
        plotting.plot(self.b, with_knots=False)
        plotting.plot(self.b, with_knots=True)



class Test_BSpline3(unittest.TestCase):
    def setUp(self):
        ex2 = {
        'control_points': np.array([[1.,2,0], [2,3,1], [2,5,3], [1,6,3], [1,9,2]]),
        'knots': np.array([1.,2.,3.,4.,5.,6.,7.])
        }
        self.b = BSpline(**ex2)

    def test_call(self):
        self.b(3.5)

    def test_scalar_shape(self):
        self.assertEqual(np.shape(self.b(3.5)), (3,))


import os

class TestDemo(unittest.TestCase):
    @unittest.skip("Testing the demo notebook")
    def test_demo(self):
        import nbformat
        from nbconvert.preprocessors.execute import ExecutePreprocessor
        here = os.path.dirname(__file__)
        demo = os.path.join(here,'Demo.ipynb')
        nb = nbformat.read(demo, as_version=4)
        pp = ExecutePreprocessor()
        pp.allow_errors = False
        pp.preprocess(nb, resources={})

class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.control_points = np.array([
            np.identity(3),
            np.array([[0.0,1,0],
                      [-1,0,0],
                      [0,0,1]]),
            np.array([[1.0,0,0],
                      [0, 0, -1],
                      [0,1,0]]),
            np.identity(3)])
        self.b1 = Spline(self.control_points)

    def test_call(self):
        self.b1(.5)

    def test_geometry(self):
        self.bg = Spline(self.control_points[1:], geometry=geometry.SO3_geometry())
        mat = self.bg(.5)
        npt.assert_allclose(np.dot(mat, mat.T), np.identity(3), atol=1e-15)
        npt.assert_allclose(self.bg(0), self.control_points[1])

    def test_geo_vectorize(self):
        self.bg = Spline(self.control_points[1:], geometry=geometry.SO3_geometry())
        mats = self.bg(np.linspace(0,.5,10))
        npt.assert_allclose(mats[0], self.control_points[1])


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.control_points = np.array([
            np.array([1,0]),
            np.array([1j, 0]),
            np.array([0, 1j]),
            np.array([0, 1])
            ])
        self.b1 = Spline(self.control_points[0:], geometry=geometry.Sphere_geometry())

    def test_call(self):
        self.b1(.5)

    def test_geometry(self):
        self.bg = Spline(self.control_points[0:], geometry=geometry.Sphere_geometry())
        v = self.bg(.85)
        npt.assert_allclose(np.inner(v, v.conj()), 1., atol=1e-15)
        npt.assert_allclose(self.bg(0), self.control_points[0])

    def test_geo_vectorize(self):
        self.bg = Spline(self.control_points[0:], geometry=geometry.Sphere_geometry())
        timesample=np.linspace(0,0.5,10)
        pts = self.bg(timesample)
        npt.assert_allclose(pts[0], self.control_points[0])
        npt.assert_allclose(np.linalg.norm(pts, axis=1), np.ones(timesample.shape))

    def test_stable_geodesic(self):
        SG = geometry.Sphere_geometry()
        P1 = self.control_points[0]
        P = SG.geodesic(P1, P1, .5)
        npt.assert_allclose(P1, P)

    def test_trivial_bezier(self):
        P = self.control_points[0]
        control_points = [P]*3
        geo = geometry.Sphere_geometry()
        b = Spline(control_points, geometry=geo)
        npt.assert_allclose(b(.5), P)

    @unittest.skip("Wasn't able to make this work with new structure")
    def test_geodesic(self):
        """
        This test is not optimal, ideally, it would compare the two geodesic functions directly, without computing any splines.
        """
        SG = geometry.Sphere_geometry()
        self.bg1 = Spline(self.control_points[0:], geometry=SG)
        v1 = self.bg1(np.linspace(.2,.4,10))
        self.bg2 = Spline(self.control_points[0:], geometry=sphere_geodesic_unstable) #This call will fail
        v2 = self.bg2(np.linspace(.2,.4,10))
        npt.assert_allclose(v1, v2)
    @unittest.skip("syntax not supported, see next test")
    def test_sp1_failed(self):
        """
        Test for 1-sphere that fails.
        """
        P = np.array([1, (1+1j)*np.sqrt(0.5), 1j, -1])
        b = Spline(P, geometry=geometry.Sphere_geometry())
        npt.assert_allclose(np.linalg.norm(b(.5)), 1.0)

    def test_sp1(self):
        """
        Test for 1-sphere that succeeds
        """
        P = np.array([[1], [(1+1j)*np.sqrt(0.5)], [1j], [-1]])
        b = Spline(P, geometry=geometry.Sphere_geometry())
        npt.assert_allclose(np.linalg.norm(b(.5)), 1.0)


class TestCP(unittest.TestCase):
    def setUp(self):
        self.control_points = np.array([
            np.array([1,0]),
            np.array([0, -1]),
            np.array([1j, 0]),
            np.array([0, 1j])
            ])
        self.b1 = Spline(self.control_points[0:], geometry=geometry.CP_geometry())

    def test_call(self):
        self.b1(.5)

    def test_stable_geodesic(self):
        P1 = self.control_points[0]
        CG = geometry.CP_geometry()
        P = CG.geodesic(P1, P1, .5)
        npt.assert_allclose(np.inner(P1.conj(),P)*P,P1) # Test for complex colinearity


    def test_geometry(self):
        self.bg = Spline(self.control_points[0:], geometry=geometry.CP_geometry())
        v = self.bg(.5)
        npt.assert_allclose(np.linalg.norm(v), 1., atol=1e-15)
        npt.assert_allclose(np.inner(self.control_points[0].conj(),self.bg(0))*self.bg(0), self.control_points[0]) # Test for complex colinearity

    def test_geo_vectorize(self):
        self.bg = Spline(self.control_points[0:], geometry=geometry.CP_geometry())
        timesample=np.linspace(0,0.5,10)
        pts = self.bg(timesample)
        npt.assert_allclose(np.inner(self.control_points[0].conj(),self.bg(0))*self.bg(0), self.control_points[0]) # Test for complex colinearity
        npt.assert_allclose(np.linalg.norm(pts, axis=1), np.ones(timesample.shape))
