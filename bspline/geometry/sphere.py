import numpy as np

from . import Geometry, sinc


class Sphere(Geometry):
    def __init__(self):
        self.type = 'sphere'

    @classmethod
    def on_manifold(self, P):
        return np.sum(np.square(P), axis=1), 1

    def geodesic(self, P1, P2, theta):
        """
        Geodesic on the 2n+1-sphere, embedded in C^(n+1)
        """
        # will not work properly for 1-sphere in C^1.
        if np.ndim(P1)==1:
            angle = np.array([np.arccos(np.clip(np.inner(P1.conj(), P2).real, -1,1))])
        else:
            angle = np.arccos(np.clip(np.einsum('ij...,ij...->i...',P1.conj(), P2).real, -1,1))
            angle = angle[:, np.newaxis,...]

        return ((1-theta)*sinc((1-theta)*angle)*P1 + theta*sinc(theta*angle)*P2)/sinc(angle)

    def involution(self, P1, P2):
        """
        The mirroring of P2 through P1
        """
        if np.ndim(P1)==1:
            return 2*np.inner(P1.conj(), P2).real*P1-P2
        else:
            return 2*np.einsum('ij...,ij...->i...', P1.conj(), P2).real*P1-P2

    def involution_derivative(self, P1, P2,V2):
        """
        The derivative of involution(P1, P2) with respect to P2, in the direction of V2
        """
        return self.involution(P1,V2)

    def exp(self, P1, V1):
        """
        Riemannian exponential
        """
        angle = np.linalg.norm(V1)
        return np.cos(angle)*P1+sinc(angle)*V1

    def log(self, P1, P2):
        """
        Riemannian logarithm
        """
        angle = np.arccos(np.clip(np.inner(P1.conj(), P2).real, -1,1))
        return (P2-np.cos(angle)*P1)/sinc(angle) #Warning: non-stable.

    def g(self, angle):
        """
        function appearing in dexpinv: (cot(theta)-1/theta)/sin(theta)
        """
        gg = np.zeros_like(angle)
        idx = np.abs(angle) < 2.0e-4
        gg[idx] = -1.0/3 - 7.0/90*angle[idx]*angle[idx] # Taylor approximation for small angles (maybe unnecessary)
        gg[~idx] = (1.0/np.tan(angle[~idx])-1.0/angle[~idx])/np.sin(angle[~idx])
        return gg

    def dexpinv(self, P1,V1,W2):
        """
        (d exp_P1)^-1_V1 (W2)
        """
        angle = np.linalg.norm(V1)
        s = np.inner(P1.conj(),W2).real #
        return (W2-s*P1)/sinc(angle)+s*self.g(angle)*V1

