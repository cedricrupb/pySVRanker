import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import trange
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import lil_matrix

_divider_ = [':', '_', '/', '\\', '-->', '->']


def is_pairwise(kernel):
    try:
        X = lil_matrix(np.zeros((2, 1)))
        kernel(X, X)
        return True
    except Exception:
        return False


def is_absolute(kernel):
    try:
        X = lil_matrix(np.zeros((2, 2)))
        kernel(X)
        return True
    except Exception:
        return False


def split_string(S):
    divider = None
    for d in _divider_:
        if d in S:
            divider = d
            break
    if divider is None:
        return [S]

    return S.split(divider)


def make_chain(preprocessors, kernel):

    def chain(X):
        for P in preprocessors:
            X = P(X)

        return kernel(X)

    return chain


def select_kernel(kernel_type):
    if kernel_type == 'euclidean':
        return euclidean
    elif kernel_type == 'jaccard':
        return jaccard_kernel
    elif kernel_type == 'rbf':
        return rbf_kernel
    elif kernel_type == 'linear':
        return linear_kernel
    else:
        raise ValueError('Unknown kernel %s' % kernel_type)


def select_preprocessor(preprocessor_type):
    if preprocessor_type == 'tfidf':
        return tfidf_preprocessor
    else:
        raise ValueError('Unknown preprocessor %s' % preprocessor_type)


def select_full(full_type):
    types = split_string(full_type)
    kernel = select_kernel(types[-1])

    if is_pairwise(kernel) and len(types) > 1:
        raise ValueError('Preprocessor are not appliable with pairwise kernels')

    preprocessors = []

    while len(types) > 1:
        act = types[0]
        types = types[1:]
        preprocessors.append(select_preprocessor(act))

    return make_chain(preprocessors, kernel)


def linear_kernel(X):
    return X.dot(X.transpose())


def _jaccard_between(X, Y):

    min_sum = X.minimum(Y).sum(axis=1, dtype=np.float64)
    max_sum = X.maximum(Y).sum(axis=1, dtype=np.float64)

    return min_sum / max_sum


def jaccard_kernel(X):
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


def tfidf_preprocessor(X):
    return TfidfTransformer().fit_transform(X)
