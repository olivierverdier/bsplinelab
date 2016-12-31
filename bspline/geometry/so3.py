import numpy as np
from . import Geometry

class SO3(Geometry):

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
