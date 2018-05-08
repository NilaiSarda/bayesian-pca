import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import decomposition as dc

'''
We will (generally) follow three papers in this implementation:
1)  Bishop et al. (1999a) [Gaussian approximation]
2)  Zhang et al. (2004)   [Reversible jump MCMC]
3)  Bishop et al. (1999b) [Variational inference]
'''

class PCA(object):
    def __init__(self, data):
        self.p = dc.PCA()
        self.data = data

    def fit_transform(self):
        self.p.fit(self.data)
        self.w = self.p.fit_transform(self.data)
        return self.w

    def log_likelihood(self):
        return self.p.score(self.data)