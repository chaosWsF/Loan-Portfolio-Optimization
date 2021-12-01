import numpy as np
from numpy.random import default_rng


class loanPortfolio:
    def __init__(self, func, seed=None):
        self.seed = seed
        self.risk_measure = func
    
    def vasicek(self, rho, length, T=1):
        seed1, seed2 = default_rng(self.seed).integers(low=1, high=999, size=2)
        W_T = default_rng(seed1).normal(0, T)
        B_T = default_rng(seed2).normal(0, T, length)
        V_T = rho * W_T + np.sqrt(1 - rho * rho) * B_T
        return V_T
    
    def default_indicator(self, rho, c, T=1):
        n = len(c)
        V_T = self.vasicek(rho, n, T)
        L_T = (V_T < c).astype(int)
        return L_T

    def asset_value(self, weights, rho, c, T=1):
        L_T = self.default_indicator(rho, c, T)
        return np.dot(weights, L_T)
    
    def risk(self, weights, rho, c, T=1):
        measure = self.risk_measure
        asset = self.asset_value(weights, rho, c, T=1)
        return measure(asset)


if __name__ == '__main__':
    rho = 0.5    # correlation
    T = 1    # maturity
