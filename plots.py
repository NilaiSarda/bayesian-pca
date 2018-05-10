import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
import sklearn.datasets as ds
import numpy as np
from vbpca import VBPCA
from lbpca import LBPCA, Coordinator
from pca import PCA

def plot_scatter(x, classes, ax=None):
    ax = plt.gca() if ax is None else ax
    cmap = plt_cm.jet
    norm = plt_col.Normalize(vmin=np.min(classes), vmax=np.max(classes))
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)
    ax.scatter(x[0, :], x[1, :], color=colors, s=20)

def plot_grid(n, ncols=4, size=(5, 5)):
    nrows = int(np.ceil(n/float(ncols)))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size[0]*ncols, size[1]*nrows))
    ax = ax.ravel()
    return [fig, ax]

def plot_bppca(y, y_classes, maxit=7, *args, **kwargs):
    np.random.seed(0)
    bppca = VBPCA(y, *args, **kwargs)
    fig, ax = plot_grid(maxit + 1)
    plot_scatter(bppca.transform(), y_classes, ax[0])
    for i in range(maxit):
        bppca.update()
        j = i + 1
        plot_scatter(bppca.transform(), y_classes, ax[j])
        ax[j].set_title('Iteration {}'.format(j))
    return bppca

def create_distributed(data, M):
    size = int(data.shape[0]/M)
    # print('batch size:', size)
    nodes = []
    for i in range(M):
        node = LBPCA(data[i*size:(i+1)*size])
        nodes.append(node)
    coord = Coordinator(data, M, nodes)
    return coord

def hinton(W, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    matrix = W
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

class GaussianDataset(object):

    def __init__(self, stdev, N):
        d = len(stdev)
        data = np.zeros((N, d))
        for i in range(N):
            for j in range(d):
                data[i, j] = np.random.normal(0, stdev[j])
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
        iris = ds.load_iris()
        self._data = iris.data
        self._shape = iris.data.shape

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

def show_hinton_weights(data):
    np.set_printoptions(precision=3)
    lbpca = LBPCA(data)
    pca = PCA(data)
    # LBPCA
    iterations = 50
    lbpca.fit_transform(iterations)
    weight = lbpca.W
    hinton(weight)
    figure = plt.gcf()
    figure.canvas.set_window_title('LBPCA, iterations=' + str(iterations))
    plt.title('LBPCA Hinton Diagram')
    plt.show()
    # PCA
    weight = pca.fit_transform()
    pcs = pca.params
    hinton(pcs[:-1,:].T)
    figure = plt.gcf()
    figure.canvas.set_window_title('PCA')
    plt.title('PCA Hinton Diagram')
    plt.show()
    # Distributed LBPCA (randomized nodes)
    iterations = 1
    coord = create_distributed(data, 10)
    coord.randomized_fit(iterations)
    weight = coord.W
    hinton(weight)
    figure = plt.gcf()
    figure.canvas.set_window_title('Distributed LBPCA (Random), iterations=' + str(iterations))
    plt.title('Distributed LBPCA (Randomized Ordering) Hinton Diagram')
    plt.show()
    # Distributed LBPCA (cyclic ordering)
    coord = create_distributed(data, 10)
    coord.cyclic_fit(iterations)
    weight = coord.W
    hinton(weight)
    figure = plt.gcf()
    figure.canvas.set_window_title('Distributed LBPCA (Cyclic), iterations=' + str(iterations))
    plt.title('Distributed LBPCA (Cyclic Ordering) Hinton Diagram')
    plt.show()
    # Distributed LBPCA (averaged updates)
    iterations = 50
    coord = create_distributed(data, 10)
    coord.averaged_fit(iterations)
    weight = coord.W
    hinton(weight)
    figure = plt.gcf()
    figure.canvas.set_window_title('Distributed LBPCA (Averaged), iterations=' + str(iterations))
    plt.title('Distributed LBPCA (Averaged Updates) Hinton Diagram')
    plt.show()

if __name__ == '__main__':
    stdev = [5, 4, 3, 2, 1, 1, 1, 1, 1, 1]
    d = GaussianDataset(stdev, 100)
    show_hinton_weights(d.data)
