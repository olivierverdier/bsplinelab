from .riemann import Riemann

class Symmetric(Riemann):
    def transport(self, P, V, W):
        return self.geometry.Adexpinv(P, V, W)
