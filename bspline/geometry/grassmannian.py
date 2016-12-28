import numpy as np

from . import Geometry, sinc, sinhc

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

    def random_direction(self, size):
        i,j = size
        p = np.random.randn(i,j)
        p = np.linalg.qr(p)[0]
        v = np.array([4])
        while np.linalg.norm(v, 2)>=0.5* np.pi:
            v = np.random.randn(i,j)/(100*np.sqrt(i))
            v = v-p.dot(p.T).dot(v) # norm(v)> pi might cause problems.
        return p, v
