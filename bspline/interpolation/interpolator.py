import numpy as np

from ..geometry import Flat
from .. import BSpline
from .boundary import make_boundaries


def cubic_spline(InterpolationClass, interpolation_points, boundaries=(None, None), geometry=Flat()):
    I = InterpolationClass(interpolation_points, make_boundaries(*boundaries), geometry)
    return I.compute_spline()

class Interpolator():
    max_iter = 500
    tolerance = 1e-12

    def __init__(self, interpolation_points, boundaries, geometry=Flat()):
        self.interpolation_points = interpolation_points
        self.boundaries = boundaries
        self.geometry = geometry
        self.size = len(self.interpolation_points)
        [boundary.initialize(self) for boundary in self.boundaries]
        self.postmortem = {}

    def compute_controls(self):
        """
        Main fixed point algorithm.
        """
        velocities = np.zeros_like(self.interpolation_points)
        for iter in range(self.max_iter):
            [boundary.enforce(velocities) for boundary in self.boundaries]
            qRs, qLs, delta = self.increment(velocities)
            velocities[1:-1] += delta
            error = np.max(np.abs(delta))
            if error < self.tolerance:
                break
        self.postmortem['error'] = error
        self.postmortem['iterations'] = iter
        return qRs, qLs

    def compute_spline_control_points(self, qRs, qLs):
        """
        Produces a spline control points from the given control points
        in an array.
        """
        geo_shape = np.shape(self.interpolation_points[0])
        new_shape = (3*self.size-2,) + geo_shape
        all_points = np.zeros(new_shape)
        all_points[::3] = self.interpolation_points
        all_points[1::3] = qRs
        all_points[2::3] = qLs
        return all_points

    def get_knots(self):
        knots = np.arange(self.size, dtype='f').repeat(3)
        return knots

    def compute_spline(self):
        """
        Produces a spline object.
        """
        qRs, qLs = self.compute_controls()
        spline_control_points = self.compute_spline_control_points(qRs, qLs)
        return BSpline(control_points=spline_control_points,
                       knots=self.get_knots(),
                       geometry=self.geometry)

    def increment(self, velocities):
        raise NotImplementedError()
