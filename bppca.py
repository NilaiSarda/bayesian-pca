import numpy as np
from sklearn import datasets

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

    def fit_gaussian(self, data, iter=100):
        N, d = data.shape
        q = d-1
        mu = np.mean(data, axis=0)
        W = np.random.normal(0,1,size=(d, q))
        sigma = 0.1
        alpha = np.random.randn(q)
        M = np.matmul(W.T, W) + sigma * np.eye(q)
        x = [None for i in range(N)]
        xx = [None for i in range(N)]
        for _ in range(iter):
            # We update the moments of x_n
            for i in range(N):
                x[i] = np.matmul(np.matmul(np.linalg.inv(M), W.T), data[i]-mu).reshape(-1,1)
                xx[i] = (sigma*M + np.matmul(x[i], x[i].T))
            A = np.diag(alpha)
            # print(x[i].T.shape)
            W = np.matmul(sum([np.matmul((data[i]-mu).reshape(-1,1), x[i].T) for i in range(N)]), np.linalg.inv(sum(xx)+sigma*A))
            sigma = (1.0/(N*d)) * sum([np.linalg.norm(data[i]-mu)**2 - np.matmul(2*np.matmul(x[i].T,W.T),data[i]-mu) + np.trace(np.matmul(xx[i], np.matmul(W.T, W))) for i in range(N)])
            M = np.matmul(W.T, W) + sigma * np.eye(q)
            for i in range(q):
                alpha[i] = d / (np.linalg.norm(W[:,i])+0.0001)
            A = np.diag(alpha)
            W = np.matmul(sum([np.matmul((data[i]-mu).reshape(-1,1), x[i].T) for i in range(N)]), np.linalg.inv(sum(xx)+sigma*A))
            sigma = (1.0/(N*d)) * sum([np.linalg.norm(data[i]-mu)**2 - np.matmul(2*np.matmul(x[i].T,W.T),data[i]-mu) + np.trace(np.matmul(xx[i], np.matmul(W.T, W))) for i in range(N)])
            M = np.matmul(W.T, W) + sigma * np.eye(q)
        self.W = W
        self.sigma = sigma
        self.alpha = alpha
        self.mu = mu
        self.N = N
        self.d = d
        self.q = q

    def likelihood(self, data):
        mu = self.mu
        W = self.W
        sigma = self.sigma
        d = self.d
        N = self.N
        q = self.q
        alpha = self.alpha
        S = 1.0/N * sum([np.outer(data[i]-mu, data[i]-mu) for i in range(N)])
        print(S)
        C = np.matmul(W,W.T) + sigma*np.eye(d)
        print(C)
        L = -1.0/2 * (d * np.log(2*np.pi) + np.log(np.sum(C)) + np.trace(np.matmul(np.linalg.inv(C), S)))
        print("L",np.log(np.sum(C)),np.trace(np.matmul(np.linalg.inv(C), S)),d * np.log(2*np.pi))
        F = -1.0/(2*N) * sum([alpha[i] * np.linalg.norm(W[i])**2 for i in range(q)])
        return (L + F)/N


class GaussianDataset(object):

    def __init__(self, stdev, N):
        d = len(stdev)
        data = np.zeros((N, d))
        for i in range(N):
            for j in range(d):
                data[i, j] = np.random.normal(1, stdev[j])
        self._data = data
        self._shape = (N, d)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


class IrisDataset(object):

    def __init__(self):
        iris = datasets.load_iris()
        self._data = iris.data
        self._shape = iris.data.shape

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape


stdev = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
d = GaussianDataset(stdev, 100)
b = BPPCA('gaussian')
b.fit(d.data)
print([np.linalg.norm(b.W[:,i]) for i in range(b.q)])
print(b.likelihood(d.data))

# d = IrisDataset()
# b = BPPCA('gaussian')
# b.fit(d.data)
# print(b.likelihood(d.data))
