import numpy as np

def sinc(x):
    """
    Unnormalized sinc function.
    """
    return np.sinc(x/np.pi)

class Geometry:
    def __init__(self):
        self.type ='flat'
        
    def geodesic(self,P1, P2, theta):
        """
        The geodesic between two points.
        """
        return (1-theta)*P1 + theta*P2
        

        

class Sphere_geometry(Geometry):
    def __init__(self):
        self.type = 'sphere'
    def geodesic(self, P1, P2, theta):
        """
        Geodesic on the 2n+1-sphere, embedded in C^(n+1)
        """
        # will not work properly for 1-sphere in C^1.
        if np.ndim(P1)==1: 
            angle = np.array([np.arccos(np.inner(P1.conj(), P2).real)])
        else:
            angle = np.arccos(np.einsum('ij...,ij...->i...',P1.conj(), P2).real)
            angle = angle[:, np.newaxis,...]
        
        return ((1-theta)*sinc((1-theta)*angle)*P1 + theta*sinc(theta*angle)*P2)/sinc(angle)
        

        

   
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
        return super().geodesic(P1, np.exp(-1j*rotations)*P2, theta)


     
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
        angles = np.arccos((Utr-1)/2) # (K,1,1,T) angle of rotation
        yhat = 0.5*(U-np.einsum('ijk...->ikj...',U)) # transpose 2. and 3. dimension.
        yhatsq =np.einsum('ijk...,ikl...->ijl...', yhat, yhat) #yhat*yhat
        invnormx = 1/np.sin(angles)
        scalar1 = np.sin(theta*angles)*invnormx #(K,1,1,T)
        scalar2 = 2*np.sin(theta*angles/2)**2*invnormx**2
        I = np.identity(3)
        I.shape = I.shape + time_shape
        V = I + scalar1*yhat + scalar2*yhatsq
        return np.einsum('ijk...,ikl...->ijl...', P1, V)




