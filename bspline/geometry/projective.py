import numpy as np
from . import Sphere
from . import  sinc

class Projective(Sphere):
    def __init__(self):
        self.type = 'complex projective plane'
    def geodesic(self,P1,P2,theta):
        if np.ndim(P1)==1:
            rotations = np.angle(np.inner(P1.conj(), P2))
            theta=np.expand_dims(theta, axis=0)
        else:
            innerprods = np.einsum('ij...,ij...->i...',P1.conj(), P2)
            rotations=np.angle(innerprods)
            rotations = rotations[:, np.newaxis,...]
        return super(Projective, self).geodesic(P1, np.exp(-1j*rotations)*P2, theta)

    def exp(self, P1, V1):
        V1hor = V1+1j*np.inner(P1.conj(), V1).imag*P1
        return super(Projective, self).exp(P1, V1hor)

    def log(self, P1, P2):
        rotations = np.angle(np.inner(P1.conj(), P2))
        return super(Projective, self).log(P1, np.exp(-1j*rotations)*P2)

    def dexpinv(self, P1, V1, W2):
        alpha = np.linalg.norm(V1)
        Wt = W2 - 1j*np.inner(P1.conj(), W2).imag*(P1+sinc(alpha)/np.cos(alpha)*V1)
        return super(Projective, self).dexpinv(P1, V1, Wt) # Assumes V1 is horizontal over P1, and that W2 is a vector over exp(P1,V1)

    def projection(self, P1):
        """
        Projection onto unique coordinate space.
        """
        return 1j*np.einsum('...i,...j->...ij', P1, P1.conj())
