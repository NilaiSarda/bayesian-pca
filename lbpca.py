import numpy as np

class LBPCA(object):

    def __init__(self, data):
        self.data = data
        self.N = self.data.shape[0]
        self.d = self.data.shape[1]
        self.q = self.d - 1
        self.mu = np.mean(self.data, axis=0)
        self.W = np.random.randn(self.d, self.q)
        self.sigma = 0
        self.alpha = np.random.randn(self.q)
        for i in range(self.q):
            self.alpha[i] = self.d / (np.linalg.norm(self.W[:,i])**2)

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
            sigma = (1.0/(N*d)) * sum([np.linalg.norm(data[i]-mu)**2 - 2*np.matmul(np.matmul(x[i].T,W.T),data[i]-mu) + np.trace(np.matmul(xx[i], np.matmul(W.T, W))) for i in range(N)])
            M = np.matmul(W.T, W) + sigma * np.eye(q)
            # update alpha
            for i in range(q):
                alpha[i] = d / (np.linalg.norm(W[:,i])**2+0.000001)
        self.W = W
        self.sigma = sigma
        self.alpha = alpha
        self.x_mean = x

    def forward(self, other):
        other.W = self.W
        other.sigma = self.sigma
        other.alpha = self.alpha
        return other.W

    def transform(self, q):
        t_W = np.array(sorted(self.W.T, key=lambda r:np.linalg.norm(r), reverse=True)).T
        return np.matmul(self.data,t_W[:,:q])

    def fit_transform(self, iterations=50):
        self.fit(iterations)
        return np.dot(self.data, self.W)

    def add(self, other):
        other.sigma += self.sigma
        other.W += self.W
        other.alpha += self.alpha

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

    def transform_infers(self):
        y = (np.array(self.x_mean).reshape(self.N, -1)).dot(self.W.T) + self.mu[:, np.newaxis].T
        return y


    def mse(self):
        d = self.data - self.transform_infers()
        d = d.ravel()
        return self.N**-1 * d.dot(d)

class Coordinator(object):

    def __init__(self, data, M, nodes):
        self.data = data
        self.N = self.data.shape[0]
        self.batch_mses = []
        self.M = M
        self.nodes = nodes

    def randomized_fit(self, iterations=50):
        data, M = self.data, self.M
        size = int(data.shape[0]/M)
        passer = LBPCA(data)
        for _ in range(iterations):
            np.random.shuffle(data)
            for i in range(M):
                node = LBPCA(data[i*size:(i+1)*size])
                passer.forward(node)
                node.fit()
                self.nodes[i] = node
                self.W = node.forward(passer)
                self.batch_mses.append(self.M*node.mse())


    def averaged_fit(self, iterations=50):
        data, M = self.data, self.M
        size = int(data.shape[0]/M)
        passer = LBPCA(data)
        accumulator = LBPCA(data)
        for _ in range(iterations):
            accumulator.W = np.zeros((accumulator.d, accumulator.q))
            accumulator.sigma = 0
            accumulator.alpha = np.zeros(accumulator.q)
            np.random.shuffle(data)
            for i in range(M):
                node = LBPCA(data[i*size:(i+1)*size])
                passer.forward(node)
                node.fit()
                self.nodes[i] = node
                node.add(accumulator)
            accumulator.W /= M
            accumulator.sigma /= M
            accumulator.alpha /= M
            self.W = accumulator.forward(passer)

    def cyclic_fit(self, iterations=50):
        nodes = list(self.nodes)
        for _ in range(iterations):
            for i in range(len(nodes)-1):
                nodes[i].fit()
                nodes[i].forward(nodes[i+1])
            nodes[-1].fit()
            self.W = nodes[-1].forward(nodes[0])

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

    def transform(self, y, q):
        t_W = np.array(sorted(self.W.T, key=lambda r:np.linalg.norm(r), reverse=True)).T
        return np.matmul(y,t_W[:,:q])

    def transform_infers(self):
        x_means = []
        mus = []
        for node in self.nodes:
            x_means.append(node.x_mean)
            mus.append(node.mu)
        self.x_mean = np.concatenate(x_means)
        self.mu = sum(mus)/len(mus)
        return (self.x_mean.reshape(self.N, -1)).dot(self.W.T) + self.mu[:, np.newaxis].T


    def mse(self):
        d = self.data - self.transform_infers()
        d = d.ravel()
        return self.N**-1 * d.dot(d)

    def get_batch_mses(self):
        return self.batch_mses
