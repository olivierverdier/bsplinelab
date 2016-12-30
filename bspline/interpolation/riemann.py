import numpy as np
from .interpolator import Interpolator

class Riemann(Interpolator):
    def generate_controls(self, points, velocities):
        for P, V in zip(points, velocities):
            yield self.geometry.exp(P, V)

    def generate_logs(self, q1s, q2s):
        for q1, q2 in zip (q1s, q2s):
            yield self.geometry.log(q1, q2)

    def transport(self, P, V, W):
        return self.geometry.dexpinv(P, V, W)

    def increment(self, velocities):
        qRs = list(self.generate_controls(self.interpolation_points[:-1], velocities[:-1]))
        qLs = list(self.generate_controls(self.interpolation_points[1:], -velocities[1:]))
        wRs = self.generate_logs(qRs[1:], qLs[1:])
        wLs = self.generate_logs(qLs[:-1], qRs[:-1])
        gen = zip(
            self.interpolation_points[1:-1],
            velocities[1:-1],
            wRs,
            wLs,
        )
        delta = np.zeros_like(velocities[1:-1])
        for i, (P, V, wR, wL) in enumerate(gen):
            delta[i] = self.transport(P, V, wR) - self.transport(P, -V, wL) - 2*V
        return qRs, qLs, delta/4
