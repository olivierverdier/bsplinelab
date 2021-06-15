from padexp import Exponential

Exp = Exponential(order=16)

def exponential(xi):
    return Exp(xi)[0]

class Geometry(object):

    def geodesic(self,P1, P2, theta):
        """
        The geodesic between two points.
        """
        raise NotImplementedError()

    def involution(self, P1, P2):
        """
        The mirroring of P2 through P1
        """
        raise NotImplementedError()

    def involution_derivative(self, P1, P2,V2):
        """
        The derivative of involution(P1, P2) with respect to P2, in the direction of V1
        """
        raise NotImplementedError()

    def exp(self, P1, V1):
        """
        Riemannian exponential
        """
        raise NotImplementedError()

    def log(self, P1, P2):
        """
        Riemannian logarithm
        """
        raise NotImplementedError()

    def dexpinv(self, P1, V1, W2):
        """ (d exp_P1)^-1_V1 (W2) """
        raise NotImplementedError()

    def allclose(self, P1, P2, *args, **kwargs):
        return np.allclose(P1, P2, *args, **kwargs)

    @classmethod
    def on_manifold(self, P):
        """
        Test if the given point is on the manifold.
        Return two arrays that should be equal to one another.
        """
        raise NotImplementedError()

    def redexp(self, P, V):
        """
        Default implementation of reduced exponential.
        """
        return exponential(self.connection(P, V))

    @classmethod
    def connection(self, P, V):
        """
        Map a velocity at point P to a deformation.
        """
        raise NotImplementedError()

    def action(self, M, P):
        """
        Action of group element M on point P.
        Default implementation is matrix-vector multiplication
        """
        return np.dot(M, P)

    def projection(self, P1):
        """
        Projection onto unique coordinate space.
        """
        return P1


import numpy as np

def sinc(x):
    """
    Unnormalized sinc function.
    """
    return np.sinc(x/np.pi)

from numpy.core.numeric import where
def sinhc(x):
    x = np.asanyarray(x)
    y = where(x == 0, 1.0e-20, x)
    return np.sinh(y)/y

from .flat import Flat
from .sphere import Sphere
from .hyperbolic import Hyperbolic
from .grassmannian import Grassmannian
from .so3 import SO3
from .projective import Projective
