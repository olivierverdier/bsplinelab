
class BoundaryCondition(object):
    def initialize(self, interpolator):
        self.interpolator = interpolator

    def enforce(self, velocities):
        """
        Enforce boundary condition in place.
        """
        velocities[self.position] = self.get_boundary_velocity(velocities)

class Free(BoundaryCondition):
    def get_boundary_velocity(self, velocities):
        neighbour = self.position + self.direction
        geo = self.interpolator.geometry
        g = geo.redexp(self.interpolator.interpolation_points[neighbour], -self.direction*velocities[neighbour])
        vel = self.direction*.5*geo.log(
            self.interpolator.interpolation_points[self.position],
            geo.action(g, self.interpolator.interpolation_points[neighbour])
        )
        return vel

class FreeL(Free):
    direction = 1
    position = 0

class FreeR(Free):
    direction = -1
    position = -1


class Clamped(BoundaryCondition):
    def __init__(self, velocity):
        self.boundary_velocity = velocity/3

    def get_boundary_velocity(self, velocities):
        return self.boundary_velocity


class ClampL(Clamped):
    position = 0

class ClampR(Clamped):
    position = -1

def get_boundary(boundary, cls0, cls1):
    if boundary is None:
        return cls0()
    else:
        return cls1(boundary)

def make_boundaries(boundaryL, boundaryR):
    return (
        get_boundary(boundaryL, FreeL, ClampL),
        get_boundary(boundaryR, FreeR, ClampR)
    )
