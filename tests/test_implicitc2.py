import numpy.testing as npt
import unittest

import numpy as np

from bspline import geometry
from bspline.c2spline import implicitc2spline

def bis(f, t, h):
    return (f(t+2*h)
            + f(t)
            - 2*f(t+h))/(h*h)

def generate_diffs(f, t, emin=2, emax=6):
    for k in range(emin,emax):
        h = 10.**(-k)
        bbis_diff = bis(f,t,h) - bis(f,t,-h)
        yield k, bbis_diff

def gen_log10_errors(f, t):
    for k,d in generate_diffs(f, t):
        err = np.log10(np.max(np.abs(d)))
        yield k, err

class HarnessImplicitC2(object):
    def setUp(self):
        N = 8
        #interpolation_points = np.array([[1.,0,0], [0,0,1.], [0, np.sqrt(0.5), np.sqrt(0.5)], [0,1.,0]]) #spline interpolates these points
        x = np.linspace(-1, 1, N)
        y = np.sin(5*np.pi*x)
        interpolation_points = np.array([x, y, np.ones(x.shape)]).T
        interpolation_points = interpolation_points/np.array([np.linalg.norm(interpolation_points, axis=1)]).T
        self.interpolation_points = interpolation_points
        #initial and end velocities:
        init_vel = np.array([-1.0,0.0,-1.0])
        end_vel = np.array([-1.0,0.0,1.0])
        boundary_velocities = np.array([init_vel, end_vel])
        self.boundary_velocities = boundary_velocities
        b = implicitc2spline(interpolation_points, boundary_velocities, geometry=self.get_geometry())
        self.b = b

    def test_interpolate(self):
        for i,P in enumerate(self.interpolation_points):
            npt.assert_allclose(self.b(i), P)

    def test_maxiter(self):
        with self.assertRaises(Exception):
            b = implicitc2spline(self.interpolation_points, self.boundary_velocities, geometry=self.get_geometry(), Maxiter=2)

    def test_c2(self, margin=.5):
        errs = np.array(list(gen_log10_errors(self.b, 1.5))).T
        imax = np.argmin(errs[1])
        emax = errs[0,imax] # maximum h exponent at a regular point
        err = errs[1,imax] + margin # expected error

        for t in range(1, len(self.interpolation_points)-1): # the joint times
            errs = np.array(list(gen_log10_errors(self.b, t))).T
            self.assertLessEqual(errs[1,imax], err)

class TestImplicitC2Flat(HarnessImplicitC2, unittest.TestCase):
    def get_geometry(self):
        return geometry.Geometry()

class TestImplicitC2Sphere(HarnessImplicitC2, unittest.TestCase):
    def get_geometry(self):
        return geometry.Sphere_geometry()

    def test_on_sphere(self, N=40):
        max = len(self.interpolation_points) - 1
        for t in max*np.random.rand(N):
            npt.assert_allclose(np.sum(np.square(self.b(t))), 1.)


