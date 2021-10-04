import numpy as np
from . import Geometry, sinhc

class Hyperbolic(Geometry):
    """
    Hyperboloid
    """

    def geodesic(self, P1, P2, theta):
        if np.ndim(P1)==1:
             arg =np.array([np.arccosh(2*P1[0]*P2[0]-np.inner(P1, P2))])
        else:
            arg = np.arccosh(2*P1[:,0]*P2[:,0]-np.einsum('ij...,ij...->i...',P1, P2))
            arg = arg[:, np.newaxis,...]
        return ((1-theta)*sinhc((1-theta)*arg)*P1 + theta*sinhc(theta*arg)*P2)/sinhc(arg)
    def exp(self, P1, V1):
        """
        Riemannian exponential
        """
        arg =np.sqrt(np.linalg.norm(V1)**2-2*V1[0]**2)
        return np.cosh(arg)*P1+sinhc(arg)*V1

    def log(self, P1, P2):
        arg = np.arccosh(2*P1[0]*P2[0]-np.inner(P1, P2))
        return (P2-np.cosh(arg)*P1)/sinhc(arg) #Warning: non-stable.

    def g(self, arg):
        """
        function appearing in dexpinv: (cot(theta)-1/theta)/sin(theta)
        """
        gg = np.zeros_like(arg)
        idx = np.abs(arg) < 2.0e-4
        gg[idx] = 1.0/3 - 7.0/90*arg[idx]*arg[idx] # Taylor approximation for small angles (maybe unnecessary)
        gg[~idx] = (1.0/np.tanh(arg[~idx])-1.0/arg[~idx])/np.sinh(arg[~idx])
        return gg

    def dexpinv(self, P1, V1, W2):
        arg = np.sqrt(np.linalg.norm(V1)**2-2*V1[0]**2)
        s = np.inner(P1,W2)-2*P1[0]*W2[0]
        return (W2+s*P1)/sinhc(arg)+s*self.g(arg)*V1

    @classmethod
    def connection(self, P, V):
        cross = V.reshape(-1,1)*P
        A = cross -cross.T
        A[:,1:]=-A[:,1:]
        return A

    def random_direction(self, size, rng=None):
        rng = np.random.default_rng(rng)
        pp = rng.standard_normal(size)
        p = np.insert(pp, 0, np.sqrt(1+np.linalg.norm(pp)**2))
        vv = rng.standard_normal(size)
        v = np.insert(vv, 0, np.inner(pp,vv)/p[0])
        return p, v

    def on_manifold(self, Ps):
        Ps2 = np.square(Ps)
        constraint = Ps2[:, 0] - np.sum(Ps2[:, 1:], axis=1)
        return constraint, 1.

