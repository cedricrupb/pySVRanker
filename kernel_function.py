import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import trange


def jaccard_kernel(X, Y):
    min_sum = np.minimum(X, Y).sum()
    max_sum = np.maximum(X, Y).sum()

    return min_sum / max_sum


def linear_kernel(X, Y):
    return np.dot(np.transpose(X), Y)


def select_kernel(kernel_type):
    if kernel_type == 'euclidean':
        return euclidean
    elif kernel_type == 'jaccard':
        return jaccard_kernel
    elif kernel_type == 'rbf':
        return rbf_kernel
    elif kernel_type == 'linear':
        return linear_kernel


def _jaccard_between(X, Y):

    min_sum = X.minimum(Y).sum(axis=1, dtype=np.float64)
    max_sum = X.maximum(Y).sum(axis=1, dtype=np.float64)

    return min_sum / max_sum


def jaccard(X):
    row = list(range(X.shape[0]))
    columns = list(range(X.shape[0]))
    data = [1] * X.shape[0]

    for i in trange(1, X.shape[0]):
        X_index = np.arange(i, X.shape[0])
        Y_index = np.arange(0, X.shape[0] - i)
        X_slice = X[X_index, :]
        Y_slice = X[Y_index, :]
        S = _jaccard_between(X_slice, Y_slice)
        S = list(S.flat)
        row.extend(X_index.data)
        columns.extend(Y_index.data)
        data.extend(S)
        row.extend(Y_index.data)
        columns.extend(X_index.data)
        data.extend(S)

    return coo_matrix((data, (row, columns))).toarray()
