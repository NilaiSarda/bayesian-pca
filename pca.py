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

    def fit(self):
        q = self.data.shape[1] - 1
        self.p.fit(self.data, n_components=q)

    def fit_transform(self):
        self.p.fit(self.data)
        self.w = self.p.fit_transform(self.data)
        return self.w

    def transform(self, y):
        self.fit()
        o = self.p.transform(y)
        return o

    def log_likelihood(self):
        return self.p.score(self.data)

    @property
    def params(self):
        return self.p.components_.T * np.sqrt(self.p.explained_variance_)
