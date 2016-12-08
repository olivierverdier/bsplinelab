class Geometry(object):
    def __init__(self):
        self.type ='flat'

    def geodesic(self,P1, P2, theta):
        """
        The geodesic between two points.
        """
        return (1-theta)*P1 + theta*P2

    def involution(self, P1, P2):
        """
        The mirroring of P2 through P1
        """
        return 2*P1-P2

    def involution_derivative(self, P1, P2,V2):
        """
        The derivative of involution(P1, P2) with respect to P2, in the direction of V1
        """
        return -V2

    def exp(self, P1, V1):
        """
        Riemannian exponential
        """
        return P1+V1

    def log(self, P1, P2):
        """
        Riemannian logarithm
        """
        return P2-P1

    def dexpinv(self, P1, V1, W2):
        """ (d exp_P1)^-1_V1 (W2) """
        return W2

    @classmethod
    def on_manifold(self, P):
        """
        Test if the given point is on the manifold.
        Return two arrays that should be equal to one another.
        """
        return 0, 0

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

from .sphere import Sphere_geometry
from .hyperbolic import Hyperboloid_geometry
from .grassmannian import Grassmannian
from .so3 import SO3_geometry
from .projective import CP_geometry
