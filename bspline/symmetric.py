
import numpy as np

from .geometry import Geometry
from . import BSpline

from padexp import Exponential

Exp = Exponential(order=16)

def exponential(xi):
    return Exp(xi)[0]

class Interpolator():
    max_iter = 500
    tolerance = 1e-12
    geometry = Geometry()

    def __init__(self, interpolation_points, boundary_velocities, geometry):
        self.interpolation_points = interpolation_points
        self.boundary_deformations = [geometry.connection(self.interpolation_points[j], boundary_velocities[i]) for i,j in ((0,0), (1,-1))]
        self.geometry = geometry

    def compute_deformations(self):
        """
        Compute the control points giving a C2 de Casteljau spline.
        """
        deformations = self.geometry.zero_deformations(self.interpolation_points)
        for i in range(self.max_iter):
            interior = self.interior_deformations(deformations)
            error = deformations[1:-1] - interior
            deformations[1:-1] = interior
            for i,j in ((0,0), (1,-1)):
                deformations[j] = self.boundary_deformations[i]
            if np.max(np.abs(error)) < self.tolerance:
                break
        else:
            raise Exception("No convergence in {} steps; error :{} ".format(i, error))
        return deformations

    def control_points(self, deformations):
        """
        Compute the control points, from the given deformations.
        """
        pass

    def interior_deformations(self, deformations):
        """
        Compute new deformations at interior points from old deformations.
        """
        N = len(deformations)

        g = self.geometry

        interior_deformations = deformations[1:-1]
        sig_left = np.zeros_like(interior_deformations)
        sig_right = np.zeros_like(interior_deformations)

        for i in range(1,N-1):
            left = g.action(exponential(-deformations[i+1]), self.interpolation_points[i+1]) # left control point at i+1
            pt_left = g.action(exponential(-deformations[i]), left)
            sig_right[i-1] = g.redlog(self.interpolation_points[i], pt_left)

            right = g.action(exponential(deformations[i-1]), self.interpolation_points[i-1]) # right control point at i-1
            pt_right = g.action(exponential(deformations[i]), right)
            sig_left[i-1] = -g.redlog(self.interpolation_points[i], pt_right)

        return (sig_left + sig_right + 2*interior_deformations)/4

