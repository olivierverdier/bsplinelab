#!/usr/bin/env python
# −*− coding: UTF−8 −*−
from __future__ import division

import numpy.testing as npt
import unittest

import numpy as np

from bspline.knots import Knots, get_basis_knots
from bspline.spline import BSpline
from bspline import plotting
from bspline.bezier import Bezier
from bspline import geometry

def sphere_geodesic_unstable(P1,P2,theta):
    """
    Geodesic on the 2n+1-sphere, undefined when P1 == P2
    """
    angle = np.arccos(np.einsum('ij...,ij...->i...',P1.conj(), P2).real)
    angle = angle[:, np.newaxis,...]
    return (np.sin((1-theta)*angle)*P1 + np.sin(theta*angle)*P2)/np.sin(angle)


def get_canonical_knots(n):
    knots = np.arange(3*n) - (3*n-1)/2
    knots[:n-1] = knots[n-1]
    knots[-(n-1):] = knots[-n]
    return Knots(knots, degree=n)


def get_basis(n):
    nb_pts = 2*n+1
    knots = np.arange(3*n) - (3*n-1)/2
    control_points = np.vstack([np.arange(nb_pts),np.zeros(nb_pts)]).T
    control_points[n,1] = 1.

    spline = BSpline(knots, control_points)
    return spline

class TestBasis(unittest.TestCase):
    def test_nonuniform(self):
        a,b,c = 0., 2.5, 8
        ck = BSpline(*get_basis_knots([a,b,c]).get_basis_data())
        npt.assert_allclose(ck(a, lknot=0)[1], 0)
        npt.assert_allclose(ck(b, lknot=1)[1], 1.)
        npt.assert_allclose(ck(c, lknot=1)[1], 0.)

    def test_constant_abscissae(self):
        k = get_basis_knots(np.arange(2))
        k.abscissae()


    ## def test_canonical(self):
    ##      ck = get_canonical_knots(5)
    ##      cb = ck.get_basis()
    ##      k = get_basis_knots(np.arange(5) - 2)
    ##      kb = k.get_basis()
    ##      b = get_basis(5)
    ##      npt.assert_allclose(cb(0.), b(0.))


class TestBezier(unittest.TestCase):
    def setUp(self):
        controls = [[1.,1],[0,-1],[-1,1]]
        self.b = Bezier(controls)

    def test_quad(self):
        u"""
    Check that Bézier with three points generates the parabola y=x**2.
        """
        b = self.b
        self.assertEqual(b.knots.nb_curves,1)
        ts = np.linspace(0.,1., 200)
        all_pts = b(ts)
        npt.assert_array_almost_equal(all_pts[:,0]**2, all_pts[:,1])
        npt.assert_allclose(b(.5), 0.)

    def test_intervals(self):
        intervals = list(self.b.knots.intervals())
        self.assertEqual(len(intervals), self.b.knots.nb_curves)

    def test_left_knot(self):
        self.assertEqual(self.b.knots.left_knot(.2), 1)
        self.assertEqual(self.b.knots.left_knot(.8), 1)

    def test_plot(self):
        plotting.plot(self.b)


class Test_DoubleQuad(unittest.TestCase):
    def setUp(self):
        controls = [[-1,1],[0,-1],[2.,3],[3,1]]
        knots = [0,0,.5,1,1]
        self.spline = BSpline(knots, controls)

    def test_info(self):
        self.assertEqual(self.spline.knots.degree, 2)
        self.assertEqual(self.spline.knots.nb_curves,2)
        self.assertEqual(len(self.spline.knots.knot_range()), self.spline.knots.nb_curves)

    def test_generate(self):
        intervals = list(self.spline.knots.intervals())
        self.assertEqual(len(intervals), self.spline.knots.nb_curves)
        a0,a1 = [self.spline(np.linspace(l,r,200)) for (k,l,r) in intervals]
        npt.assert_array_almost_equal(a0[:,0]**2, a0[:,1])
        npt.assert_array_almost_equal(-(a1[:,0]-2)**2, a1[:,1]-2)

class Test_BSpline(unittest.TestCase):
    def setUp(self):
        ex2 = {
        'control_points': np.array([[1.,2], [2,3], [2,5], [1,6], [1,9]]),
        'knots': np.array([1.,2.,3.,4.,5.,6.,7.])
        }
        self.b = BSpline(**ex2)

    def test_left_knot(self):
        self.assertEqual(self.b.knots.left_knot(3.8), 2)
        self.assertEqual(self.b.knots.left_knot(3.2), 2)
        self.assertEqual(self.b.knots.left_knot(4.8), 3)
        self.assertEqual(self.b.knots.left_knot(4.0), 3)
        self.assertEqual(self.b.knots.left_knot(4.0-1e-14), 3)
        with self.assertRaises(ValueError):
            self.b.knots.left_knot(2.5)
        with self.assertRaises(ValueError):
            self.b.knots.left_knot(5.5)

    def test_knot_range(self):
        k = Knots(np.arange(10))
        self.assertEqual(len(k.knot_range()), 0)

    def test_wrong_knot(self):
        with self.assertRaises(ValueError):
            self.b(3.5, lknot=1)
        with self.assertRaises(ValueError):
            self.b(3.5, lknot=4)

    def test_abscissae(self):
        pts = np.random.random_sample([7,2])
        knots = [0,0,0,2,3,4,5,5,5]
        b = BSpline(knots, pts)
        computed = b.knots.abscissae()
        expected = np.array([0, 2/3, 5/3, 3, 4, 14/3, 5]) # values from Sederberg §6.14
        npt.assert_allclose(computed, expected)

    def test_plot(self):
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

class TestKnots(unittest.TestCase):
    def test_basis(self):
        """
        Check that the basis functions sum up to one.
        """
        w = [ 0, 0, 0, 1/3, 2/3, 1, 1, 1]
        wk = Knots(w, degree=3)
        basis = [BSpline(*wk.get_basis_data(i)) for i in range(6)]
        for k,l,r in wk.intervals():
            ts = np.linspace(l,r,30)
            vals = np.array([b(ts, lknot=k) for b in basis])
            npt.assert_allclose(np.sum(vals[:,:,1], axis=0), 1.)

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
        self.b1 = Bezier(self.control_points)

    def test_call(self):
        self.b1(.5)

    def test_geometry(self):
        self.bg = Bezier(self.control_points[1:], geometry=geometry.SO3_geometry())
        mat = self.bg(.5)
        npt.assert_allclose(np.dot(mat, mat.T), np.identity(3), atol=1e-15)
        npt.assert_allclose(self.bg(0), self.control_points[1])

    def test_geo_vectorize(self):
        self.bg = Bezier(self.control_points[1:], geometry=geometry.SO3_geometry())
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
        self.b1 = Bezier(self.control_points[0:], geometry=geometry.Sphere_geometry())
    
    def test_call(self):
        self.b1(.5)
        
    def test_geometry(self):
        self.bg = Bezier(self.control_points[0:], geometry=geometry.Sphere_geometry())
        v = self.bg(.85)
        npt.assert_allclose(np.inner(v, v.conj()), 1., atol=1e-15)
        npt.assert_allclose(self.bg(0), self.control_points[0])   

    def test_geo_vectorize(self):
        self.bg = Bezier(self.control_points[0:], geometry=geometry.Sphere_geometry())
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
        b = Bezier(control_points, geometry=geo)
        npt.assert_allclose(b(.5), P)
        
    @unittest.skip("Wasn't able to make this work with new structure")
    def test_geodesic(self):
        """
        This test is not optimal, ideally, it would compare the two geodesic functions directly, without computing any splines.
        """
        SG = geometry.Sphere_geometry()
        self.bg1 = Bezier(self.control_points[0:], geometry=SG)
        v1 = self.bg1(np.linspace(.2,.4,10))
        self.bg2 = Bezier(self.control_points[0:], geometry=sphere_geodesic_unstable) #This call will fail
        v2 = self.bg2(np.linspace(.2,.4,10))
        npt.assert_allclose(v1, v2)
    @unittest.skip("syntax not supported, see next test")
    def test_sp1_failed(self):
        """
        Test for 1-sphere that fails.
        """
        P = np.array([1, (1+1j)*np.sqrt(0.5), 1j, -1])
        b = Bezier(P, geometry=geometry.Sphere_geometry())
        npt.assert_allclose(np.linalg.norm(b(.5)), 1.0)
        
    def test_sp1(self):
        """
        Test for 1-sphere that succeeds
        """
        P = np.array([[1], [(1+1j)*np.sqrt(0.5)], [1j], [-1]])
        b = Bezier(P, geometry=geometry.Sphere_geometry())
        npt.assert_allclose(np.linalg.norm(b(.5)), 1.0)
                

class TestCP(unittest.TestCase):
    def setUp(self):
        self.control_points = np.array([
            np.array([1,0]),
            np.array([0, -1]),
            np.array([1j, 0]),
            np.array([0, 1j])
            ])
        self.b1 = Bezier(self.control_points[0:], geometry=geometry.CP_geometry())
    
    def test_call(self):
        self.b1(.5)
            
    def test_stable_geodesic(self):
        P1 = self.control_points[0]
        CG = geometry.CP_geometry()
        P = CG.geodesic(P1, P1, .5)
        npt.assert_allclose(np.inner(P1.conj(),P)*P,P1) # Test for complex colinearity 

        
    def test_geometry(self):
        self.bg = Bezier(self.control_points[0:], geometry=geometry.CP_geometry())
        v = self.bg(.5)
        npt.assert_allclose(np.linalg.norm(v), 1., atol=1e-15)
        npt.assert_allclose(np.inner(self.control_points[0].conj(),self.bg(0))*self.bg(0), self.control_points[0]) # Test for complex colinearity 
        
    def test_geo_vectorize(self):
        self.bg = Bezier(self.control_points[0:], geometry=geometry.CP_geometry())
        timesample=np.linspace(0,0.5,10)
        pts = self.bg(timesample)
        npt.assert_allclose(np.inner(self.control_points[0].conj(),self.bg(0))*self.bg(0), self.control_points[0]) # Test for complex colinearity 
        npt.assert_allclose(np.linalg.norm(pts, axis=1), np.ones(timesample.shape))
        
