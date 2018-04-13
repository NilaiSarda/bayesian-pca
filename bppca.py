import numpy as np
import scipy
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

    def fit(self, data, iterations):
        if self.method == 'gaussian':
            self.fit_gaussian(data, iterations)
            self.likelihood = self.gaussian_likelihood
        elif self.method == 'gibbs':
            self.fit_gibbs(data, iterations)
            self.likelihood = self.gibbs_likelihood
        elif self.method == 'vb':
            self.fit_vb(data)
        elif self.method == 'jump':
            self.fit_jump(data)
        else:
            print("Try a valid method, please")

    def fit_gaussian(self, data, iterations):
        N, d = data.shape
        q = d - 1
        mu = np.mean(data, axis=0)
        W = np.random.normal(0,1,size=(d, q))
        sigma = 0.1
        alpha = np.random.randn(q)
        M = np.matmul(W.T, W) + sigma * np.eye(q)
        x = [None for i in range(N)]
        xx = [None for i in range(N)]
        for _ in range(iterations):
            # We update the moments of x_n
            for i in range(N):
                x[i] = np.matmul(np.matmul(np.linalg.inv(M), W.T), data[i]-mu).reshape(-1,1)
                xx[i] = (sigma*M + np.matmul(x[i], x[i].T))
            A = np.diag(alpha)
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

    def gaussian_likelihood(self, data):
        mu = self.mu
        W = self.W
        sigma = self.sigma
        d = self.d
        N = self.N
        q = self.q
        alpha = self.alpha
        print('|W_i|:', [np.linalg.norm(W[:,i]) for i in range(q)])
        # S = 1.0/N * sum([np.outer(data[i]-mu, data[i]-mu) for i in range(N)])
        # print('S:', S)
        # C = np.matmul(W,W.T) + sigma*np.eye(d)
        # print('C:', C)
        # L = N/-2 * (d * np.log(2*np.pi) + np.log(np.sum(np.abs(C))) + np.trace(np.matmul(np.linalg.inv(C), S)))
        # print('L:', N/-2, '*', '(' + str(d * np.log(2*np.pi)), '+', np.log(np.sum(np.abs(C))), '+', str(np.trace(np.matmul(np.linalg.inv(C), S))) + ')')
        # F = -1.0/2 * sum([alpha[i] * np.linalg.norm(W[:, i])**2 for i in range(q)])
        # print('F:', F)
        C = np.matmul(W,W.T) + sigma*np.eye(d)
        print('C:', C)
        L = sum([1.0/-2 * (np.matmul(np.matmul((data[i]-mu).T, np.linalg.inv(C)), (data[i]-mu)) + np.log((2*np.pi) ** d * np.linalg.det(C))) for i in range(N)])
        print('L:', L)
        return L/N

    def fit_gibbs(self, data, iterations):
        N, d = data.shape
        q = np.random.randint(1, d)
        alpha = 1.0
        eta = 0.5
        r = 1.0
        tau = np.random.gamma(alpha, eta)
        prior_samples = np.sort(np.random.gamma(r, tau, q + 1))
        l_inv = np.zeros(d-1)
        l_inv[:q] = prior_samples[:-1]
        variance_inv = prior_samples[-1]
        mu = np.mean(data, axis=0)
        S = 1.0/N * sum([np.outer(data[i]-mu, data[i]-mu) for i in range(N)])
        A, g, vh = np.linalg.svd(S)
        b = np.ones(d)/2.0
        b[1] = 1
        b[d-1] = 0
        de = np.ones(d)/2.0
        de[1] = 0
        de[d-1] = 1
        l_inv_sum = np.zeros(d-1)
        variance_inv_sum = 0
        for _ in range(iterations):
            for i in range(q):
                l_inv[i] = np.random.gamma(N/2 + r, N*g[i]/2 + tau)
                if (i > 0 and l_inv[i] < l_inv[i-1]) or (i < d-2 and l_inv[i] > l_inv[i+1]):
                    l_inv[i] = 0
            variance_inv = np.random.gamma(N*(d-q)/2 + r, N*np.sum(g[q:])/2 + tau)
            if variance_inv < l_inv[q-1]:
                variance_inv = np.random.gamma(N*(d-q)/2 + r, N*np.sum(g[q:])/2 + tau)
            tau = np.random.gamma((q+1)*r + alpha, np.sum(l_inv[:q]) + variance_inv + eta)
            log_likelihood = lambda q : (-N*d/2) * np.log(2*np.pi) + (N/2) * sum([np.log(l_inv[j] + 0.00001) for j in range(q)]) + (N*(d-q)*q/2) * np.log(variance_inv + 0.00001) + (-N/2) * sum([l_inv[j] * g[j] for j in range(q)]) + (-N*variance_inv/2) * sum([g[j] for j in range(q, d)])
            u = np.random.uniform()
            if u <= b[q]:  # birth move
                l_inv_q_prev = l_inv[q]
                # formula at top of page 3
                before_log_likelihood = log_likelihood(q)
                l_inv[q] = np.random.gamma(r, tau)
                if l_inv[q] < l_inv[q-1] or l_inv[q] > variance_inv:
                    l_inv[q] = 0
                log_R = log_likelihood(q+1) - before_log_likelihood + np.log(q+2) + np.log(de[q+1]) - np.log(b[q])
                print('log_R:', log_R)
                v = np.random.uniform()
                if np.log(v) >= log_R:  # reject update
                    l_inv[q] = l_inv_q_prev
                else:
                    q = q + 1
            else:  # death move
                l_inv_qless_prev = l_inv[q-1]
                before_log_likelihood = log_likelihood(q)
                l_inv[q-1] = 0
                log_R_inv = log_likelihood(q-1) - before_log_likelihood - np.log(q+1) - np.log(de[q]) + np.log(b[q-1])
                print('log_R_inv:', log_R_inv)
                v = np.random.uniform()
                if np.log(v) >= log_R_inv:  # reject update
                    l_inv[q-1] = l_inv_qless_prev
                else:
                    q = q - 1
            l_inv_sum += l_inv
            variance_inv_sum += variance_inv
        self.l_inv = l_inv_sum / iterations
        self.variance_inv = variance_inv_sum / iterations
        self.g = g
        self.N = N
        self.d = d

    def gibbs_likelihood(self, data):
        l_inv = self.l_inv
        variance_inv = self.variance_inv
        g = self.g
        N = self.N
        d = self.d
        np.seterr(divide='ignore')
        print('l_i:', 1.0/l_inv)
        print('variance:', 1.0/variance_inv)
        log_likelihood = lambda q : (-N*d/2) * np.log(2*np.pi) + (N/2) * sum([np.log(l_inv[j] + 0.00001) for j in range(q)]) + (N*(d-q)*q/2) * np.log(variance_inv + 0.00001) + (-N/2) * sum([l_inv[j] * g[j] for j in range(q)]) + (-N*variance_inv/2) * sum([g[j] for j in range(q, d)])
        for q in range(1, d):
            print('L(' + str(q) + '):', log_likelihood(q))


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
b.fit(d.data, 500)
print(b.likelihood(d.data))
print()
print()
b = BPPCA('gibbs')
b.fit(d.data, 50)
b.likelihood(d.data)
