import numpy as np
from .interpolator import Interpolator

class Symmetric(Interpolator):
    def generate_controls(self, points, velocities):
        """
        Generate movements and control points.
        """
        for p, v in zip(points, velocities):
            g = self.geometry.redexp(p, v)
            control = self.geometry.action(g, p)
            yield g, control

    def increment(self, velocities):
        gRs, qRs = list(zip(*self.generate_controls(self.interpolation_points[:-1], velocities[:-1])))
        gLs, qLs = list(zip(*self.generate_controls(self.interpolation_points[1:], -velocities[1:])))
        delta = np.zeros_like(velocities[1:-1])
        gen = zip(
            self.interpolation_points[1:-1],
            velocities[1:-1],
            gLs[:-1],
            qLs[1:],
            gRs[1:],
            qRs[:-1],
        )
        for i, (p, v, gL, qL, gR, qR) in enumerate(gen):
            delta[i] = self.geometry.log(p, self.geometry.action(gL, qL)) - self.geometry.log(p, self.geometry.action(gR, qR)) - 2*v
        return qRs, qLs, delta/4





