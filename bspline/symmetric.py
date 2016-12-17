
import numpy as np

from .geometry import Flat
from . import BSpline

class BoundaryCondition(object):
    def initialize(self, interpolator):
        self.interpolator = interpolator

class Free(BoundaryCondition):
    def get_boundary_deformations(self, deformations):
        geo = self.interpolator.geometry
        defs = [s*.5*geo.redlog(self.interpolator.interpolation_points[j], geo.exp_action(self.interpolator.interpolation_points[i], -s*deformations[i])) for j,i,s in ((0,1,1),(-1,-2,-1))]
        return defs

class Clamped(BoundaryCondition):
    def __init__(self, boundary_velocities):
        self.boundary_velocities = boundary_velocities

    def initialize(self, interpolator):
        super(Clamped, self).initialize(interpolator)
        geo = interpolator.geometry
        self.boundary_deformations = [1/3*geo.connection(interpolator.interpolation_points[j], self.boundary_velocities[i]) for i,j in ((0,0), (1,-1))]

    def get_boundary_deformations(self, deformations):
        return self.boundary_deformations

def cubic_spline(interpolation_points, boundary=Free(), geometry=Flat()):
    I = Interpolator(interpolation_points, boundary, geometry)
    return I.compute_spline()

class Interpolator():
    max_iter = 500
    tolerance = 1e-12

    def __init__(self, interpolation_points, boundary=Free(), geometry=Flat()):
        self.interpolation_points = interpolation_points
        self.boundary = boundary
        self.geometry = geometry
        self.size = len(self.interpolation_points)
        self.boundary.initialize(self)

    def compute_spline(self):
        """
        Produces a spline object.
        """
        deformations = self.compute_deformations()
        control_points = np.array(list(self.control_points(deformations)))
        spline_control_points = self.compute_spline_control_points(control_points)
        knots = np.arange(len(self.interpolation_points), dtype='f').repeat(3)
        return BSpline(control_points=spline_control_points,
                       knots=knots,
                       geometry=self.geometry)


    @classmethod
    def enforce(self, deformations, boundary_deformations):
        for pos, deformation in zip([0,-1], boundary_deformations):
            deformations[pos] = deformation


    def compute_deformations(self):
        """
        Compute the control points giving a C2 de Casteljau spline.
        """
        deformations = self.geometry.zero_deformations(self.interpolation_points)
        for i in range(self.max_iter):
            interior = self.interior_deformations(deformations)
            error = deformations[1:-1] - interior
            deformations[1:-1] = interior
            self.enforce(deformations, self.boundary.get_boundary_deformations(deformations))
            max_error = np.max(np.abs(error))
            if max_error < self.tolerance:
                break
        else:
            raise Exception("No convergence in {} steps; error :{} ".format(i, error))
        self.postmortem = {'error': max_error, 'iterations': i}
        return deformations

    def control_points(self, deformations, shift=1):
        """
        Compute the interior control points, from the given deformations.
        """
        N = self.size
        all_range = range(N)
        left_range = all_range[shift:]
        right_range = all_range[:-shift]
        geo = self.geometry
        for l,r in zip(left_range, right_range):
            # left control point at i+1
            left = geo.exp_action(self.interpolation_points[l], -deformations[l])
            # right control point at i-1
            right = geo.exp_action(self.interpolation_points[r], deformations[r])
            yield right, left

    def interior_deformations(self, deformations):
        """
        Compute new deformations at interior points from old deformations.
        """
        geo = self.geometry

        interior_deformations = deformations[1:-1]
        sig_left = np.zeros_like(interior_deformations)
        sig_right = np.zeros_like(interior_deformations)

        for i, (P, d, (right, left)) in enumerate(zip(
                self.interpolation_points[1:-1],
                interior_deformations,
                self.control_points(deformations, shift=2))):
            pt_left = geo.exp_action(left, -d)
            sig_right[i] = geo.redlog(P, pt_left)
            pt_right = geo.exp_action(right, d)
            sig_left[i] = geo.redlog(P, pt_right)

        return (sig_right - sig_left + 2*interior_deformations)/4

    def compute_spline_control_points(self, control_points):
        """
        Produces a spline control points from the given control points
        in an array.
        """
        geo_shape = np.shape(self.interpolation_points[0])
        new_shape = (3*self.size-2,) + geo_shape
        all_points = np.zeros(new_shape)
        all_points[::3] = self.interpolation_points
        all_points[1::3] = control_points[:,0]
        all_points[2::3] = control_points[:,1]
        return all_points
