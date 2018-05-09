import numpy as np
import random

class LBPCA(object):

    def __init__(self, y):
        self.y = y
        self.N = self.y.shape[0]
        self.d = self.y.shape[1]
        self.q = self.d - 1

    def fit(self, data, iterations=50):
        N, d, q = self.N, self.d, self.q
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

    def transform(self):
        return np.dot(self.y, self.W)

    def fit_transform(self):
        self.fit(self.y)
        return np.dot(self.y, self.W)

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

class Node(object):

    def __init__(self, data):
        self.data = data
        self.N = self.data.shape[0]
        self.d = self.data.shape[1]
        self.q = self.d - 1
        self.mu = np.mean(self.data.data, axis=0)
        self.W = np.random.normal(0,1,size=(self.d, self.q))
        self.sigma = 0.1
        self.alpha = np.random.randn(self.q)

    def update(self, iterations=1):
        N, d, q = self.N, self.d, self.q
        mu, W, sigma, alpha = self.mu, self.W, self.sigma, self.alpha
        data = self.data
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

    def forward(self, other):
        other.sigma, other.W, other.alpha = self.sigma, self.W, self.alpha
        return self.W

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

class Coordinator(object):

    def __init__(self, data, dict):
        self.dict = dict
        self.data = data

    def randomized_fit(self, iterations=1000):
        for _ in range(iterations):
            nodes = list(self.dict.keys())
            random.shuffle(nodes)
            for i in range(len(nodes)-1):
                nodes[i].update()
                self.W = nodes[i].forward(nodes[i+1])

    def cyclic_fit(self, iterations=1000):
        nodes = list(self.dict.keys())
        for _ in range(iterations):
            for i in range(len(nodes)-1):
                nodes[i].update()
                self.W = nodes[i].forward(nodes[i+1])
