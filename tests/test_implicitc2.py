import numpy.testing as npt
import unittest

import numpy as np

from bspline import geometry
from bspline.c2spline import implicitc2spline

class TestImplicitC2(unittest.TestCase):
    def setUp(self):
        N = 8
        #interpolation_points = np.array([[1.,0,0], [0,0,1.], [0, np.sqrt(0.5), np.sqrt(0.5)], [0,1.,0]]) #spline interpolates these points
        x = np.linspace(-1, 1, N)
        y = np.sin(5*np.pi*x)
        interpolation_points = np.vstack([x,y,np.ones(x.shape)]).T
        interpolation_points = interpolation_points/np.array([np.linalg.norm(interpolation_points, axis=1)]).T
        self.interpolation_points = interpolation_points
        #initial and end velocities:
        init_vel = np.array([-1.0,0.0,-1.0])
        end_vel = np.array([-1.0,0.0,1.0])
        boundary_velocities = np.array([init_vel, end_vel])
        self.boundary_velocities = boundary_velocities
        b = implicitc2spline(interpolation_points, boundary_velocities, geometry=geometry.Sphere_geometry())
        self.b = b

    def test_on_sphere(self, N=40):
        max = len(self.interpolation_points) - 1
        for t in max*np.random.rand(N):
            npt.assert_allclose(np.sum(np.square(self.b(t))), 1.)

    def test_interpolate(self):
        for i,P in enumerate(self.interpolation_points):
            npt.assert_allclose(self.b(i), P)

    def test_maxiter(self):
        with self.assertRaises(Exception):
            b = implicitc2spline(self.interpolation_points, self.boundary_velocities, geometry=geometry.Sphere_geometry(), Maxiter=2)
