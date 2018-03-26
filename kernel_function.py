import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import rbf_kernel


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
