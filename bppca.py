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
    def __init__(self):
        self.p = dc.PCA()

    def fit(self, data):
        self.p.fit(data)
        self.w = self.p.fit_transform(data)

    def log_likelihood(self, data):
        return self.p.score(data)

    def hinton(self, max_weight=None, ax=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        matrix = self.w
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()


class BPPCA(object):

    def __init__(self, method):
        self.method = method
        self.q_dist = Q(self.n, self.p, self.q)

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
        C = np.matmul(W,W.T) + sigma*np.eye(d)
        print('C:', C)
        L = sum([1.0/-2 * (np.matmul(np.matmul((data[i]-mu).T, np.linalg.inv(C)), (data[i]-mu)) + np.log((2*np.pi) ** d * np.linalg.det(C))) for i in range(N)])
        print('L:', L)
        return L/N

    def fit_gibbs(self, data, iterations):
        N, d = data.shape
        q = np.random.randint(1, d)
        alpha = N/2
        eta = 1.0/2
        r = N/2
        tau = np.random.gamma(alpha, eta)
        print('r:', r)
        print('tau:', tau)
        prior_samples = np.sort(np.random.gamma(r, tau, q + 1))[::-1]
        l_inv = np.zeros(d-1)
        l_inv[:q] = prior_samples[1:]
        variance_inv = prior_samples[0]
        mu = np.mean(data, axis=0)
        S = 1.0/N * sum([np.outer(data[i]-mu, data[i]-mu) for i in range(N)])
        A, g, vh = np.linalg.svd(S)
        print('g[i]:', g)
        print('q:', q)
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
                print('left:', N/2 + r)
                print('right:', N*g[i]/2 + tau)
                l_inv[i] = np.random.gamma(N/2 + r, N*g[i]/2 + tau)
                while (i > 0 and l_inv[i] >= l_inv[i-1]):
                    l_inv[i] = np.random.gamma(N/2 + r, N*g[i]/2 + tau)
            print('l_inv[i]:', l_inv)
            variance_inv = np.random.gamma(N*(d-q)/2 + r, N*np.sum(g[q:])/2 + tau)
            while variance_inv <= l_inv[q-1]:
                print('left (var):', N*(d-q)/2 + r)
                print('right (var):', N*np.sum(g[q:])/2 + tau)
                variance_inv = np.random.gamma(N*(d-q)/2 + r, N*np.sum(g[q:])/2 + tau)
            tau = np.random.gamma((q+1)*r + alpha, np.sum(l_inv[:q]) + variance_inv + eta)
            log_likelihood = lambda q : (-N*d/2) * np.log(2*np.pi) + (N/2) * sum([np.log(l_inv[j] + 0.00001) for j in range(q)]) + (N*(d-q)*q/2) * np.log(variance_inv + 0.00001) + (-N/2) * sum([l_inv[j] * g[j] for j in range(q)]) + (-N*variance_inv/2) * sum([g[j] for j in range(q, d)])
            u = np.random.uniform()
            if u <= b[q]:  # birth move
                l_inv_q_prev = l_inv[q]
                # formula at top of page 3
                before_log_likelihood = log_likelihood(q)
                l_inv[q] = np.random.gamma(r, tau)
                while l_inv[q] >= l_inv[q-1] or l_inv[q] >= variance_inv:
                    l_inv[q] = np.random.gamma(r, tau)
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

    def fit_vb(self, data, n_iter=100):
        self.p = data.shape[0]
        self.n = data.shape[1]
        self.q = 2
        self.alpha_a = 1.0
        self.alpha_b = 1.0
        self.gamma_a = 1.0
        self.gamma_b = 1.0
        self.beta = 1.0

        def update_x():
            q = self.q_dist
            gamma_mean = q.gamma_a / q.gamma_b
            q.x_cov = np.linalg.inv(np.eye(self.q) + gamma_mean * np.transpose(q.w_mean).dot(q.w_mean))
            q.x_mean = gamma_mean * q.x_cov.dot(np.transpose(q.w_mean)).dot(data - q.mu_mean[:, np.newaxis])

        def update_w():
            q = self.q_dist
            # cov
            x_cov = np.zeros((self.q, self.q))
            for n in range(self.n):
                x = q.x_mean[:, n]
                x_cov += x[:, np.newaxis].dot(np.array([x]))
            q.w_cov = np.diag(q.alpha_a / q.alpha_b) + q.gamma_mean() * x_cov
            q.w_cov = np.linalg.inv(q.w_cov)
            # mean
            yc = data - q.mu_mean[:, np.newaxis]
            q.w_mean = q.gamma_mean() * q.w_cov.dot(q.x_mean.dot(np.transpose(yc)))
            q.w_mean = np.transpose(q.w_mean)

        def update_mu():
            q = self.q_dist
            gamma_mean = q.gamma_a / q.gamma_b
            q.mu_cov = (self.beta + self.n * gamma_mean)**-1 * np.eye(self.p)
            q.mu_mean = np.sum(data - q.w_mean.dot(q.x_mean), 1)
            q.mu_mean = gamma_mean * q.mu_cov.dot(q.mu_mean)

        def update_alpha():
            q = self.q_dist
            q.alpha_a = self.alpha_a + 0.5 * self.p
            q.alpha_b = self.alpha_b + 0.5 * np.linalg.norm(q.w_mean, axis=0)**2

        def update_gamma():
            q = self.q_dist
            q.gamma_a = self.gamma_a + 0.5 * self.n * self.p
            q.gamma_b = self.gamma_b
            w = q.w_mean
            ww = np.transpose(w).dot(w)
            for n in range(self.n):
                y = data[:, n]
                x = q.x_mean[:, n]
                q.gamma_b += y.dot(y) + q.mu_mean.dot(q.mu_mean)
                q.gamma_b += np.trace(ww.dot(x[:, np.newaxis].dot([x])))
                q.gamma_b += 2.0 * q.mu_mean.dot(w).dot(x[:, np.newaxis])
                q.gamma_b -= 2.0 * y.dot(w).dot(x)
                q.gamma_b -= 2.0 * y.dot(q.mu_mean)

        for _ in range(n_iter):
            update_mu()
            update_w()
            update_x()
            update_alpha()
            update_gamma()
        self.W = np.dot(self.q_dist.w_mean,self.q_dist.x_mean)

    def infer(self):
        q = self.q_dist
        x = q.x_mean
        w, mu = q.w_mean, q.mu_mean
        y = w.dot(x) + mu[:, np.newaxis]
        return y

    def transform(self, y=None):
        if y is None:
            return self.q_dist.x_mean
        q = self.q_dist
        [w, mu, sigma] = [q.w, q.mu, q.gamma**-1]
        m = np.transpose(w).dot(w) + sigma * np.eye(w.shape[1])
        m = np.linalg.inv(m)
        x = m.dot(np.transpose(w)).dot(y - mu)
        return x

    def mse(self, data):
        d = data - self.infer()
        d = d.ravel()
        return self.n**-1 * d.dot(d)

    def hinton(self, max_weight=None, ax=None):
        matrix = self.W
        """Draw Hinton diagram for visualizing a weight matrix."""
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()




class Q(object):
    def __init__(self, n, p, q):
        self.n = n
        self.p = p
        self.q = q
        self.init()

    def init(self):
        self.x_mean = np.random.normal(0.0, 1.0, self.q * self.n).reshape(self.q, self.n)
        self.x_cov = np.eye(self.q)
        self.w_mean = np.random.normal(0.0, 1.0, self.p * self.q).reshape(self.p, self.q)
        self.w_cov = np.eye(self.q)
        self.alpha_a = 1.0
        self.alpha_b = np.empty(self.q)
        self.alpha_b.fill(1.0)
        self.mu_mean = np.random.normal(0.0, 1.0, self.p)
        self.mu_cov = np.eye(self.p)
        self.gamma_a = 1.0
        self.gamma_b = 1.0

    def gamma_mean(self):
        return self.gamma_a / self.gamma_b

    def alpha_mean(self):
        return self.alpha_a / self.alpha_b


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


# stdev = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
# d = GaussianDataset(stdev, 10)
# p = PCA()
# p.fit(d.data)
# print(p.log_likelihood(d.data))
# print(p.p.components_)
# print()
# p.hinton()
# plt.show()
# b = BPPCA('gaussian')
# b.fit(d.data, 50)
# print(b.likelihood(d.data))
# print(b.W)
# print()
# b.hinton()
# plt.show()
# b = BPPCA('vb')
# b.fit(d.data, 20)
# print(b.mse(d.data))
# print(b.q_dist.w_mean)
# b.hinton()
# plt.show()
# b = BPPCA('gibbs')
# b.fit(d.data, 50)
# b.likelihood(d.data)
