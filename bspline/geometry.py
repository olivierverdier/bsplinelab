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

class Sphere_geometry(Geometry):
    def __init__(self):
        self.type = 'sphere'

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


class CP_geometry(Sphere_geometry):
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
        return super(CP_geometry, self).geodesic(P1, np.exp(-1j*rotations)*P2, theta)

    def exp(self, P1, V1):
        V1hor = V1+1j*np.inner(P1.conj(), V1).imag*P1
        return super(CP_geometry, self).exp(P1, V1hor)
    
    def log(self, P1, P2):
        rotations = np.angle(np.inner(P1.conj(), P2))
        return super(CP_geometry, self).log(P1, np.exp(-1j*rotations)*P2)
        
    def dexpinv(self, P1, V1, W2):
        alpha = np.linalg.norm(V1)
        Wt = W2 - 1j*np.inner(P1.conj(), W2).imag*(P1+sinc(alpha)/np.cos(alpha)*V1)
        return super(CP_geometry, self).dexpinv(P1, V1, Wt) # Assumes V1 is horizontal over P1, and that W2 is a vector over exp(P1,V1)

class Hyperboloid_geometry(Geometry):   
    def __init__(self):
        self.type = 'Hyperboloid'
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
        
class SO3_geometry(Geometry):       
    def __init__(self):
         self.type = 'SO3'
    def geodesic(self, P1,P2,theta):
        """
        Geodesics on SO3  calculated via formulas p. 363-364 in 'Lie group methods'
        Uses einsum for literally everything to handle (K,3,3) and (K,3,3,T) data.
        """
        time_shape = (1,)*(np.ndim(P1)-3) # guess the time shape from the number of dimensions of the given points
        U = np.einsum('imj...,imk...->ijk...', P1,P2)  # P1^T*P2
        Utr = np.einsum('ijj...->i...', U) #trace of U
        Utr = Utr[:, np.newaxis,np.newaxis,...]
        angles = np.arccos(np.clip((Utr-1)/2, -1,1)) # (K,1,1,T) angle of rotation
        yhat = 0.5*(U-np.einsum('ijk...->ikj...',U)) # transpose 2. and 3. dimension.
        yhatsq =np.einsum('ijk...,ikl...->ijl...', yhat, yhat) #yhat*yhat
        invnormx = 1/np.sin(angles)
        scalar1 = np.sin(theta*angles)*invnormx #(K,1,1,T)
        scalar2 = 2*np.sin(theta*angles/2)**2*invnormx**2
        I = np.identity(3)
        I.shape = I.shape + time_shape
        V = I + scalar1*yhat + scalar2*yhatsq
        return np.einsum('ijk...,ikl...->ijl...', P1, V)
        
class Grassmannian(Geometry):
    
    def __init__(self):
        self.type = 'Grassmannian'
        
    def geodesic(self,P1, P2, theta): #(K,N,M,T), (K,1,T)
        ntd = np.ndim(P1)-3
        ntt = np.ndim(theta)-2
        
        P1 = P1.transpose(list(range(3, 3+ntd))+[0,1,2]) # (T,K,N,M)
        P2 = P2.transpose(list(range(3,3+ntd))+[0,1,2])
        if(ntd<ntt):
            P1= P1[np.newaxis,...]
            P2 = P2[np.newaxis,...]
        theta= theta.transpose(list(range(2,2+ntt))+[0,1]) #(T,K,1)
        P12 = np.einsum('...ij,...ik->...jk', P1, P2) #(T,K,M,M)
        U, s, V = np.linalg.svd(P12, full_matrices=False) #(T,K,M,M), (T,K,M), (T,K,M,M)
        alpha = np.arccos(np.clip(s, -1,1))   #(T,K,M)
        VV = np.einsum('...ik,...jk', P2, V)-np.einsum('...ik,...kj,...j->...ij', P1,U,s) #(T,K,N,M)
        Y = np.einsum('...ik,...k,...jk', VV, sinc(alpha)**(-1), U)  #(T,K,N,M)
        G = np.einsum('...ij,...jk,...k->...ik', P1, U,np.cos(theta*alpha))+np.einsum('...ij,...jk,...k->...ik',Y,U,theta*sinc(theta*alpha)) #(T,K,N,M)
        G2 = np.einsum('...ij,...kj',G, U) #(T,K,N,M)
        G2 = G2.transpose(list(range(ntt, 3+ntt))+list(range(ntt)))
        if G2.shape[3] == 1:
            G2 = G2.squeeze(3)
        return G2
        
    def exp(self,P1, Y):
        U, s, V = np.linalg.svd(Y, full_matrices=False) 
        G = np.einsum('...ij,...kj,...k->...ik', P1,V,np.cos(s)) + np.einsum('...ij,...j->...ij', U, np.sin(s))
        return np.einsum('...ij, ...jk', G, V)
        
    def log(self,P1, P2):
        P12 = np.einsum('...ij,...ik->...jk', P1, P2)
        U, s, V = np.linalg.svd(P12, full_matrices=False) 
        alpha = np.arccos(np.clip(s, -1,1))
        VV = np.einsum('...ik,...jk', P2, V)-np.einsum('...ik,...kj,...j->...ij', P1,U,s)
        return  np.einsum('...ik,...k,...jk', VV, sinc(alpha)**(-1), U)

    def dexpinv(self, P1, V1, W2):
        ## Slow algorithm with NxN -matrices. Faster algorithm with NxK matrices possible?
        n, k = P1.shape
        G1 = np.linalg.qr(np.hstack((P1, np.eye(n))))[0] # expand  P1 to NxN ort. matrix
        Q1 = G1[...,:, k:]
        Y1 = (Q1.T).dot(V1)
        s, U = np.linalg.eigh(Y1.dot(Y1.T))
        s=np.sqrt(np.clip(s,0, np.inf)) #eigh can return -eps for singular matrices
        L = np.einsum('...ik, ...k, ...jk->...ij', U, np.cos(s)**(-1), U)
        W = L.dot(Q1.T).dot(W2)
        ZZ1 = np.zeros((k,k))
        ZZ2 = np.zeros((n-k, n-k))
        YY = np.vstack((np.hstack((ZZ1, -Y1.T)), np.hstack((Y1, ZZ2)))) # [0 -Y.T ; Y 0]
        WW = np.vstack((np.hstack((ZZ1, -W.T)), np.hstack((W, ZZ2)))) 
        L, V = np.linalg.eigh(1j*YY)
        L = -1j*L
        L2 = np.outer(L, np.ones_like(L))- np.outer(np.ones_like(L), L)
        A = (V.T.conj()).dot(WW).dot(V)
        B = A/sinhc(L2)
        UU = (V.dot(B).dot(V.T.conj())).real
        return Q1.dot(UU[k:,:k])
        
        