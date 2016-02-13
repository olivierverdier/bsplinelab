# coding=utf-8
import numpy as np
from . import BSpline

class Knots(object):
    """
Knots class.

If:
    P = Nb points
    C = Nb curves
    K = Nb knots
    D = Degree

then the following relations are implemented in the code:

    C = K - 2D + 1
    D = K - P + 1

This also gives

    C + D = P
    K + 1 = P + D

For example, suppose that there are P=n+1 control points:

======== ========= ====== =======
nb knots nb curves degree remarks
-------- --------- ------ -------
n        n+1       0      n+1 points
n+1      n         1      n segments
n+2      n-1       2
...      ...       ...
2n       1         n      if first n and last n knots are equal: Bézier case
-------- --------- ------ -------
    """
    def __init__(self, knots, degree=0):
        self.knots = np.array(knots, float)
        self.degree = degree

    def __repr__(self):
        return "<{} polynomials of degree {}>".format(self.nb_curves, self.degree)

    @property
    def nb_curves(self):
        return  len(self.knots) - 2*self.degree + 1

    def __getitem__(self, index):
        return self.knots[index]

    ktol = 1e-13

    def left_knot(self, t):
        """
        Find out between which node a time t is.
        """
        diff = self.knots[self.degree-1:-self.degree+1] - t
        isrightof = diff > self.ktol
        if np.all(isrightof):
            raise ValueError("Time too small")
        if np.all(~isrightof):
            raise ValueError("Time too big")
        left = np.argmax(isrightof) - 1 # argmax gives the right knot...
        return left + self.degree-1

    def abscissae(self):
        """
        Return the Greville abscissae.
        """
        if self.degree == 0:
            k = np.hstack([-1, self.knots])
            return k
        kernel = np.ones(self.degree)/self.degree
        res = np.convolve(kernel, self.knots, 'valid')
        return res

    def knot_range(self):
        """
The range of knots from which to generate the points.
        """
        if self.degree == 0:
            return []
        return range(self.degree - 1, self.degree - 1 + self.nb_curves)

    def intervals(self, knot_range=None):
        """
        Compute the intervals from knot numbers `knot_range` till the next ones.
        """
        if knot_range is None:
            knot_range = self.knot_range()
        for k in knot_range:
            left, right = self.knots[k], self.knots[k+1]
            yield (k, left, right)

    def get_basis(self, k=None):
        if k is None:
            k = self.degree
        abscissae = self.abscissae()
        pts = np.zeros([len(abscissae), 2])
        pts[:,0] = self.abscissae()
        pts[k,1] = 1.
        return BSpline(knots=self.knots, control_points=pts)


def get_basis_knots(x):
    """
    Knots corresponding to the points in the array x.
    The corresponding basis is obtained by get_basis_knots(x).get_basis()
    """
    x = np.array(x)
    degree = len(x) - 2
    knots = np.hstack([(degree-1)*[x[0]], x, (degree-1)*[x[-1]]])
    return Knots(knots, degree)
