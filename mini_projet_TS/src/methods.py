import numpy as np
from ripser import Rips
import persim
import warnings 
warnings.filterwarnings("ignore")

class SyntheticData:

    def __init__(self, 
                 w: int,
                 d: int=4) -> None:

        self.w = w
        self.d = d

    def generate_data(self):
        raise NotImplementedError("Se référer aux classes filles pour générer des données.")
    
class NoisyHenon(SyntheticData):

    def __init__(self, 
                 w: int,
                 b: np.array,
                 sigma: float,
                 x0: float,
                 y0: float,
                 n: int,
                 T: float=1.4,
                 a: float=0,
                 d: int=4) -> None:
        
        super().__init__(w, d)
        self.b = b
        self.sigma = sigma
        self.x0 = x0
        self.y0 = y0
        self.n = n
        self.T = T
        self.a = a

    def generate_data(self, indep_Gn):
        
        a_values, step = np.linspace(self.a, self.T, self.n, retstep=True)
        y = self.y0
        xs = np.full((self.n, self.d), self.x0, dtype=float)

        for i in range(1, self.n):

            xs[i] = 1 - a_values[i]*xs[i-1]**2 + self.b*y + self.sigma*np.sqrt(step)*indep_Gn[i]
            y = xs[i] + self.sigma*np.sqrt(step)*indep_Gn[i]

        return a_values, xs
    
    def compute_wasserstein_distances(self, indep_Gn):

        rips = Rips(maxdim=2)
        m = self.n - 2*self.w + 1
        wasserstein_dists = np.zeros((m, 1))
        a_values, xs = self.generate_data(indep_Gn)

        for i in range(m):
            dgm1 = rips.fit_transform(xs[i:i+self.w])
            dgm2 = rips.fit_transform(xs[i+self.w+1:i+(2*self.w)+1])

            wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0], matching=False)
        return a_values, xs, wasserstein_dists