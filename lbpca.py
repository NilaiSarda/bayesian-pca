import numpy as np

class LBPCA(object):

    def __init__(self, data):
        self.data = data
        self.N = self.data.shape[0]
        self.d = self.data.shape[1]
        self.q = self.d - 1
        self.mu = np.mean(self.data.data, axis=0)
        self.W = np.random.normal(0,1,size=(self.d, self.q))
        self.sigma = 0.1
        self.alpha = np.random.randn(self.q)

    def fit(self, iterations=1):
        data = self.data
        N, d, q = self.N, self.d, self.q
        mu, W, sigma, alpha = self.mu, self.W, self.sigma, self.alpha
        M = np.matmul(W.T, W) + sigma * np.eye(q)
        x = [None for i in range(N)]
        xx = [None for i in range(N)]
        for _ in range(iterations):
            # update the moments of x_n
            for i in range(N):
                x[i] = np.matmul(np.matmul(np.linalg.inv(M), W.T), data[i]-mu).reshape(-1,1)
                xx[i] = (sigma*M + np.matmul(x[i], x[i].T))
            # update W, sigma
            A = np.diag(alpha)
            W = np.matmul(sum([np.matmul((data[i]-mu).reshape(-1,1), x[i].T) for i in range(N)]), np.linalg.inv(sum(xx)+sigma*A))
            sigma = (1.0/(N*d)) * sum([np.linalg.norm(data[i]-mu)**2 - np.matmul(2*np.matmul(x[i].T,W.T),data[i]-mu) + np.trace(np.matmul(xx[i], np.matmul(W.T, W))) for i in range(N)])
            M = np.matmul(W.T, W) + sigma * np.eye(q)
            # update the moments of x_n
            for i in range(N):
                x[i] = np.matmul(np.matmul(np.linalg.inv(M), W.T), data[i]-mu).reshape(-1,1)
                xx[i] = (sigma*M + np.matmul(x[i], x[i].T))
            # update alpha
            for i in range(q):
                alpha[i] = d / (np.linalg.norm(W[:,i])+0.0001)
        self.W = W
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, other):
        other.W = self.W
        other.sigma = self.sigma
        other.alpha = self.alpha
        return other.W

    def transform(self):
        return np.dot(self.data, self.W)

    def fit_transform(self, iterations=50):
        self.fit(iterations)
        return np.dot(self.data, self.W)

    def add(self, other):
        other.sigma += self.sigma
        other.W += self.W
        other.alpha += self.alpha
        other.sigma /= 2
        other.W /= 2
        other.alpha /= 2
        return other.W

    def gaussian_likelihood(self):
        data = self.data
        N, d, q = self.N, self.d, self.q
        mu, W, sigma, alpha = self.mu, self.W, self.sigma, self.alpha
        print('|W_i|:', [np.linalg.norm(W[:,i]) for i in range(q)])
        C = np.matmul(W,W.T) + sigma*np.eye(d)
        print('C:', C)
        L = sum([1.0/-2 * (np.matmul(np.matmul((data[i]-mu).T, np.linalg.inv(C)), (data[i]-mu)) + np.log((2*np.pi) ** d * np.linalg.det(C))) for i in range(N)])
        print('L:', L)
        return L/N

class Coordinator(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def randomized_fit(self, iterations=50):
        for _ in range(iterations):
            nodes = list(self.nodes)
            np.random.shuffle(nodes)
            for i in range(len(nodes)-1):
                nodes[i].fit()
                self.W = nodes[i].forward(nodes[i+1])

    def cyclic_fit(self, iterations=50):
        nodes = list(self.nodes)
        for _ in range(iterations):
            for i in range(len(nodes)-1):
                nodes[i].fit()
                self.W = nodes[i].forward(nodes[i+1])

    def robust_fit(self, iterations=100):
        nodes = list(self.dict.keys())
        for _ in range(iterations):
            for i in range(len(nodes)-1):
                leader = nodes[i]
                leader.update()
                delta = np.zeros((leader.d, leader.q))
                for j in range(len(nodes)-1):
                    if j != i:
                        worker = nodes[j]
                        leader.forward(worker)
                        worker.update()
                        worker.add(leader)
        self.W = leader.W
