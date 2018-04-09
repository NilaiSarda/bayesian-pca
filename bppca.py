import numpy as np

'''
We will (generally) follow three papers in this implementation:
1)  Bishop et al. (1999a) [Gaussian approximation]
2)  Zhang et al. (2004)   [Reversible jump MCMC]
3)  Bishop et al. (1999b) [Variational inference]
'''

class BPPCA(object):

    def __init__(self, method):
        self.method = method

    def fit(self, data):
        if self.method == 'gaussian':
            self.fit_gaussian(data)
        elif self.method == 'gibbs':
            self.fit_gibbs(data)
        elif self.method == 'vb':
            self.fit_vb(data)
        elif self.method == 'jump':
            self.fit_jump(data)
        else:
            print("Try a valid method, please")

    def fit_gaussian(self, data):
        


class Dataset(object):
    def __init__(self, stdev, N):
        d = len(stdev)
        data = np.zeros((N, d))
        for i in range(N):
            for j in range(d):
                data[i,j] = np.random.normal(0,stdev[j])
        self._data = data
        self._shape = (N, d)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

stdev = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
d = Dataset(stdev, 20)
b = BPPCA('gaussian')
b.fit(d.data)
