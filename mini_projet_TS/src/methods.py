import numpy as np
from ripser import Rips
import persim
import pandas as pd
import yfinance as yf

import src.utils as ut

import warnings 
warnings.filterwarnings("ignore")

class SyntheticData:

    def __init__(self,
                 d: int=4) -> None:
        
        self.d = d

    def generate_data(self):
        raise NotImplementedError("Se référer aux classes filles pour générer des données.")
    
    
class NoisyHenon(SyntheticData):

    def __init__(self,
                 b: np.array,
                 sigma: float,
                 x0: float,
                 y0: float,
                 n: int,
                 T: float=1.4,
                 a: float=0,
                 d: int=4) -> None:
        
        super().__init__(d)
        self.b = b
        self.sigma = sigma
        self.x0 = x0
        self.y0 = y0
        self.n = n
        self.T = T
        self.a = a

    def generate_data(self, 
                      indep_Gn: np.array) -> (np.array, np.array):
        
        a_values, step = np.linspace(self.a, self.T, self.n, retstep=True)
        y = self.y0
        xs = np.full((self.n, self.d), self.x0, dtype=float)

        for i in range(1, self.n):

            xs[i] = 1 - a_values[i]*xs[i-1]**2 + self.b*y + self.sigma*np.sqrt(step)*indep_Gn[i]
            y = xs[i] + self.sigma*np.sqrt(step)*indep_Gn[i]

        return a_values, xs
    
    def compute_wasserstein_distances(self,
                                      w : int,
                                      indep_Gn: np.array,
                                      ) -> (np.array, np.array, np.array):

        rips = Rips(maxdim=2)
        m = self.n - 2*w + 1
        wasserstein_dists = np.zeros((m, 1))
        a_values, xs = self.generate_data(indep_Gn)

        for i in range(m):
            dgm1 = rips.fit_transform(xs[i:i+w])
            dgm2 = rips.fit_transform(xs[i+w+1:i+(2*w)+1])

            wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0], matching=False)
        return a_values, xs, wasserstein_dists
    

class FinancialData(SyntheticData):

    def __init__(self,
                 d: int=4) -> None:
        
        super().__init__(d)

    def generate_data(self,
                      indices: dict,
                      start_date: float,
                      end_date: float):
        
        index_data = {name: yf.download(symbol, start=start_date, end=end_date)
              for name, symbol in indices.items()}

        for _, df in index_data.items():
            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'date',
                'Adj Close': 'adj_close',
                'Volume': 'volume',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close'
            }, inplace=True)
        
        min_length = np.inf
        log_returns = []
        for _, df in index_data.items():
            log_return = ut.compute_log_returns(ut.format_dataframe(df))
            log_returns.append(log_return)
            min_length = min(min_length, len(log_return["adj_close_lr"]))

        for i in range(len(log_returns)):
            log_returns[i] = log_returns[i]["adj_close_lr"][:min_length]

        index = df["date"][:min_length]
        columns = list(index_data.keys())
        datas = np.array(log_returns).T

        data = pd.DataFrame(datas, columns = columns, index = index)

        return data
    

    def compute_landscapes(self,
                           data: pd.DataFrame,
                           w_window_size: int,
                           k_homology_dimension: int,
                           m_landscape:          int,
                           n_nodes:              int,
                           memory_saving:        tuple = (False, 1)) -> tuple:
        """
        For a given set of financial data, computes the respective persistence
        diagrams and landscapes and display the full L1 and L2 norm persistence
        landscape time series along with a more restricted visualization centered
        around the Dotcom bubble.
        """
        # Abbreviates parameters
        w   = w_window_size
        k   = k_homology_dimension
        m   = m_landscape
        n   = n_nodes
        mem = memory_saving
        # Computes landscapes
        diagrams   = ut.compute_persistence_diagrams(data, w, memory_saving=mem)
        landscapes = ut.compute_persistence_landscapes(diagrams, k, m, n, mem[0])
        # Computes norms
        norms_df   = ut.compute_persistence_landscape_norms(landscapes)

        return diagrams, landscapes, norms_df