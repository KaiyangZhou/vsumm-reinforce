import numpy as np
from scipy.spatial.distance import cdist

"""
K-medoids
Author: Kaiyang Zhou
Homepage: https://kaiyangzhou.github.io/
"""

class KMedoids(object):
    def __init__(self, n_medoids=5, distance_metric='euclidean', max_iter=500):
        self.n_medoids = n_medoids
        self.distance_metric = distance_metric
        self.max_iter = max_iter

    def fit(self, X):
        X = self._check_X(X)
        D = cdist(X, X, metric=self.distance_metric)
        n_examples = D.shape[0]

        M = np.arange(n_examples)
        np.random.shuffle(M)
        M = np.sort(M[:self.n_medoids])

        for iteration in xrange(self.max_iter):
            # assign each example to a medoid
            assignments = M[np.argmin(D[M,:], axis=0)]

            # update medoids
            M_new = np.copy(M)
            for cidx in xrange(self.n_medoids):
                m = M_new[cidx]
                idxs = [i for i in xrange(n_examples) if assignments[i] == m]
                sub_D = D[idxs,:]
                sub_D = sub_D[:,idxs]
                cost = np.sum(sub_D, axis=1)
                m_new = idxs[np.argmin(cost)]
                M_new[cidx] = m_new

            M_new = np.sort(M_new)
            if np.array_equal(M, M_new):
                M = M_new
                break

            M = M_new

        self.medoids_idxs = M
        self.medoids = X[M,:]

    def _check_X(self, X):
        n_examples = X.shape[0]

        if self.n_medoids > n_examples:
            raise Exception('Error: n_medoids is larger than n_examples')

        if not isinstance(X, np.ndarray):
            X = np.asarray(X).astype('float32')
        else:
            if X.dtype != 'float32':
                X = X.astype('float32')

        return X

    def predict(self, X):
        if X.ndim == 1:
            dim = X.shape[0]
            if dim != self.medoids.shape[1]:
                raise Exception('feature dimension of input data does not match that of medoids')
            else:
                X = X[None,:]
        elif X.ndim == 2:
            dim = X.shape[1]
            if dim != self.medoids.shape[1]:
                raise Exception('feature dimension of input data does not match that of medoids')

        D = cdist(self.medoids, X, metric=self.distance_metric)

        return np.argmin(D, axis=0)

    def get_medoids_idxs(self):
        return self.medoids_idxs

    def get_medoids(self):
        return self.medoids

if __name__ == '__main__':
    '''
    Generate random clusters to test this algorithm
    '''
    from matplotlib import pyplot as plt

    sigma = 0.1
    dev = 0.3
    n_samples = 50

    cluster1 = np.random.randn(n_samples,2) * sigma
    cluster1[:,0] += dev
    cluster1[:,1] += dev
    cluster2 = np.random.randn(n_samples,2) * sigma
    cluster2[:,0] += dev
    cluster2[:,1] -= dev
    cluster3 = np.random.randn(n_samples,2) * sigma
    cluster3[:,0] -= dev
    cluster3[:,1] -= dev
    cluster4 = np.random.randn(n_samples,2) * sigma
    cluster4[:,0] -= dev
    cluster4[:,1] += dev
    data = np.concatenate([cluster1, cluster2, cluster3, cluster4], axis=0)

    km = KMedoids(n_medoids=4, distance_metric='euclidean', max_iter=300)
    km.fit(data)
    km_idxs = km.get_medoids_idxs()

    plt.plot(data[:,0], data[:,1], 'bo')
    plt.plot(data[km_idxs,0], data[km_idxs,1], 'ro')
    plt.savefig('kmedoids_example.png')
    plt.close()
