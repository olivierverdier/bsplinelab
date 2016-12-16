import numpy as np

from . import Geometry

class Flat(Geometry):
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

    @classmethod
    def connection(self, P, V):
        """
        Map a velocity at point P to a deformation
        """
        n = len(V)
        mat = np.zeros((n+1,n+1))
        mat[:-1,-1] = V
        mat[-1,-1] = 0.
        return mat

    def action(self, M, P):
        """
        Not the simple matrix multiplication due to how we store the points.
        """
        return np.dot(M[:-1,:-1], P) + M[:-1,-1]
