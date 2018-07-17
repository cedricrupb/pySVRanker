import numpy as np
from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService
from .bag_tasks import BagFeatureTask, FeatureJsonService, BagGraphIndexTask
from .bag_tasks import BagLabelMatrixTask, index, BagFilterTask, reverse_index
from .bag_tasks import ranking
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import time
import math
from .rank_scores import select_score
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist, pdist, squareform
from .kernel_function import select_full
from .bag import normalize_gram
from scipy.sparse import vstack, coo_matrix
from scipy.stats import norm
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from tqdm import trange


__divider__ = [':', '_', '/', '\\', '-->', '->']


def may_divider(s):
    for d in __divider__:
        if d.startswith(s):
            return True
    return False


def make_chain(transformers):

    def chain(nodeIndex, X_train, y_train, X_test):
        for (t, param) in transformers:
            nodeIndex, X_train, X_test =\
                    t(nodeIndex, X_train, y_train, X_test, param)

        return nodeIndex, X_train, X_test

    return chain


def parse_transform_definition(definition):
    definition = definition+'#'
    state = 0
    buffer = ''
    param_key_buffer = ''
    param_val_buffer = ''
    params = {}
    i = 0

    transformers = []

    while i < len(definition):
        c = definition[i]
        if c == ' ':
            i += 1
        elif state == 0:
            if may_divider(c) or c == '#':
                state = 1
            elif c == '(':
                state = 3
                i += 1
            else:
                buffer += c
                i += 1
        elif state == 1:
            transformer = select_transformer(buffer)
            if transformer is None:
                raise ValueError('Unknown transformer %s' % buffer)
            transformers.append((transformer, params))
            params = {}
            buffer = ''
            state = 2
        elif state == 2:
            buffer += c
            if buffer in __divider__:
                state = 0
                buffer = ''
            i += 1
        elif state == 3:
            if c == '=':
                state = 4
            elif c == ')':
                if len(param_key_buffer) > 0:
                    raise ValueError('Unfinished param %s' % param_key_buffer)
                state = 0
            else:
                param_key_buffer += c
            i += 1
        elif state == 4:
            if c == ',' or c == ')':
                params[param_key_buffer] = param_val_buffer
                param_key_buffer = ''
                param_val_buffer = ''
                if c == ')':
                    state = 0
                else:
                    state = 3
            else:
                param_val_buffer += c
            i += 1

    return make_chain(transformers)


def select_transformer(t_id):
    if t_id == 'tfidf':
        return tfidf_transformer
    if t_id == 'tfidfc':
        return tfidf_class_transformer
    if t_id == 'density':
        return density_transformer
    if t_id == 'count':
        return count_transformer
    if t_id == 'svd':
        return count_transformer
    if t_id == 'id':
        return identity
    if t_id == 'bns':
        return bns_transformer
    if t_id == 'chi2':
        return chi2_transformer
    if t_id == 'mi':
        return mi_transformer
    if t_id == 'hsic':
        return hsic_transformer
    if t_id == 'coverage':
        return coverage_transformer


def tfidf_transformer(nodeIndex, X, y, X_test, params=None):
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    X_test = transformer.transform(X_test)
    return nodeIndex, X, X_test


def num_classes(y):
    def bin_num(row):
        sum = 0
        for i, a in enumerate(range(row.shape[0])):
            sum += row[a]*(2**(row.shape[0]-i-1))
        return sum
    return np.apply_along_axis(bin_num, 1, y)


def _document_frequency(X, y):
    N = 2**(y.shape[1])

    cat = num_classes(y)

    subjects = []
    for i in range(N):
        subjects.append(X[np.where(cat == i)[0], :].sum(axis=0))
    subjects = np.vstack(subjects)
    return N + 1, np.count_nonzero(subjects, axis=0) + 1


def tfidf_class_transformer(nodeIndex, X, y, X_test, params=None):
    if params is not None and 'cut' in params:
        y = y[:, :int(params['cut'])]
    N, freq = _document_frequency(X, y)
    idf = np.log(float(N)/freq) + 1.0
    idf = sp.spdiags(idf, diags=0, m=X.shape[1],
                     n=X.shape[1], format='csr')
    X = normalize(X * idf, norm='l2', copy=False)
    X_test = normalize(X_test * idf, norm='l2', copy=False)
    return nodeIndex, X, X_test


def density_transformer(nodeIndex, X, y, X_test, params=None):
    density = 0.99
    if params is not None and 'percent' in params:
        density = float(params['percent'])
    col_density = X.sum(axis=0)
    full_density = col_density.sum()
    col_index = col_density.argsort()[:, ::-1]
    col_density = col_density[np.unravel_index(col_index, col_density.shape)]

    i = 0
    j = col_index.shape[0]

    est = full_density*density

    while col_density[:, :j].sum() > est:
        m = int((i+j)/2)
        if col_density[:, :m].sum() <= est:
            i = m + 1
        else:
            j = m - 1

    ix = np.unravel_index(col_index[:, :j], col_density.shape)[1][0]
    return nodeIndex[ix], X[:, ix], X_test[:, ix]


def count_transformer(nodeIndex, X, y, X_test, params=None):
    count = 10000
    if params is not None and 'number' in params:
        count = int(params['number'])
    col_density = X.sum(axis=0)
    col_index = col_density.argsort()[:, ::-1][:, :count]

    ix = np.unravel_index(col_index, col_density.shape)[1][0]
    return nodeIndex[ix], X[:, ix], X_test[:, ix]


def svd_transformer(nodeIndex, X, y, X_test, params=None):
    count = 100
    if params is not None and 'number' in params:
        count = int(params['number'])
    svd = TruncatedSVD(
        n_components=count
    )

    X = svd.fit_transform(X)
    nodeIndex = np.array(['SVD_%d' % i for i in range(X.shape[1])])
    X_test = svd.transform(X_test)
    return nodeIndex, X, X_test


def identity(nodeIndex, X, y, X_test, params=None):
    return nodeIndex, X, X_test


def bns_transformer(nodeIndex, X, y, X_test, params=None):
    bns = []

    for i in range(y.shape[1]):
        y_act = y[:, i]
        rows_pos = []
        cols_pos = []
        data_pos = []
        rows_neg = []
        cols_neg = []
        data_neg = []

        for (row, col) in np.transpose(X.nonzero()):
            if y_act[row] == 1:
                rows_pos.append(row)
                cols_pos.append(col)
                data_pos.append(1)
            else:
                rows_neg.append(row)
                cols_neg.append(col)
                data_neg.append(1)
        tpr = coo_matrix((data_pos, (rows_pos, cols_pos)),
                         shape=(y_act.shape[0], X.shape[1])).tocsr()
        tpr = tpr.sum(axis=0)/y_act.sum()
        fpr = coo_matrix((data_neg, (rows_neg, cols_neg)),
                         shape=(y_act.shape[0], X.shape[1])).tocsr()
        fpr = tpr.sum(axis=0)/(np.ones(y_act.shape) - y_act).sum()

        bns.append(np.absolute(norm.ppf(tpr)-norm.ppf(fpr)))

    bns = np.vstack(bns)
    bns = bns.mean(axis=0)

    count = 10000
    if params is not None and 'count' in params:
        count = int(params['count'])

    ix = bns.argsort()[:count]
    return nodeIndex[ix], X[:, ix], X_test[:, ix]


def chi2_transformer(nodeIndex, X, y, X_test, params=None):
    count = 1000
    if params is not None:
        if 'cut' in params:
            y = y[:, :int(params['cut'])]
        if 'count' in params:
            count = int(params['count'])

    cat = num_classes(y)
    t = SelectKBest(chi2, count)
    X = t.fit_transform(X, cat)
    X_test = t.transform(X_test)
    return nodeIndex[t.get_support(indices=True)], X, X_test


def mi_transformer(nodeIndex, X, y, X_test, params=None):
    count = 1000
    if params is not None:
        if 'cut' in params:
            y = y[:, :int(params['cut'])]
        if 'count' in params:
            count = int(params['count'])

    cat = num_classes(y)
    t = SelectKBest(mutual_info_classif, count)
    X = t.fit_transform(X, cat)
    X_test = t.transform(X_test)
    return nodeIndex[t.get_support(indices=True)], X, X_test


def resolve_label(y, base):
    tmp = []

    for i in range(base-1):
        for j in range(i+1, base):
            L_i = y[:, index(i, i, base)]
            L_j = np.logical_not(y[:, index(j, j, base)])
            L_ij = y[:, index(i, j, base)]
            T2 = np.logical_or(L_i, L_j)
            T2 = np.logical_and(T2, L_ij)
            T = np.logical_and(L_i, L_j)
            T = np.logical_or(T, T2)
            tmp.append(T.reshape(T.shape[0], 1))

    return np.hstack(tmp)


def label_kernel(y):
    return np.ones((y.shape[0], y.shape[0])) - squareform(pdist(y, 'hamming'))


def hsic_transformer(nodeIndex, X, y, X_test, params=None):
    count = 1000
    if params is not None:
        if 'cut' in params:
            y = y[:, :int(params['cut'])]
        if 'count' in params:
            count = int(params['count'])
        if 'resolve' in params:
            y = resolve_label(y, int(params['resolve']))
    L = label_kernel(y)
    H = np.full(L.shape, 1/L.shape[0])
    H = np.diag(np.ones((L.shape[0]))) - H
    M = H*L*H

    F = X.copy()
    F[F > 0] = 1

    fs = np.zeros(F.shape[1])
    for i in trange(fs.shape[0]):
        fg = F[:, i]
        fs[i] = (fg.transpose())*M*fg
    print(fs)

    ix = fs.argsort()[::-1][:count]
    return nodeIndex[ix], X[:, ix], X_test[:, ix]


def coverage_transformer(nodeIndex, X, y, X_test, params=None):
    count = 1000
    X_ = X.copy()
    X_[X_ > 0] = 1
    X_ = X_.sum(axis=0) / X_.shape[0]

    if params is not None:
        if 'class' in params:
            X_ = X_ * _document_frequency(X, y)
        if 'count' in params:
            count = int(params['count'])

    ix = X_.argsort()[::-1][:count]
    return nodeIndex[ix], X[:, ix], X_test[:, ix]


class FSFeatureTransformTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, transform_expression, h, D,
                 train_index, test_index,
                 category=None, task_type=None):
        self.h = h
        self.D = D
        self.train_index = train_index
        self.test_index = test_index
        self.category = category
        self.task_type = task_type
        self.transform_expression = transform_expression

    def require(self):
        return [BagLabelMatrixTask(self.h, self.D,
                                   self.category, self.task_type),
                BagFeatureTask(self.h, self.D,
                               self.category, self.task_type)
                ]

    def __taskid__(self):
        return 'FSFeatureTransformTask_%s' % (str(
                                          containerHash(
                                                        list(
                                                             self.get_params().items()
                                                            )
                                                        )
                                           )
                                      )

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return LocalTarget(path, service=FeatureJsonService)

    def run(self):
        with self.input()[0] as inp:
            label = inp.query()

        label = np.array(label['label_matrix'])

        with self.input()[1] as inp:
            D = inp.query()

        graphIndex = D['graphIndex']
        nodeIndex = D['nodeIndex']
        features = D['features']

        train_index = np.array(self.train_index)
        test_index = np.array(self.test_index)

        X_train = features[train_index]
        label = label[train_index]
        X_test = features[test_index]

        rev_nodeIndex = np.array(
            [x[0] for x in sorted(list(nodeIndex.items()),
                                  key=lambda x: x[1])]
        )

        transformer = parse_transform_definition(self.transform_expression)
        rev_nodeIndex, X_train, X_test =\
            transformer(rev_nodeIndex, X_train, label, X_test)

        nodeIndex_new = {k: v for v, k in enumerate(rev_nodeIndex)}
        print(X_train.shape)

        with self.output() as o:
            o.emit(
                {
                    'params': self.get_params(),
                    'graphIndex': graphIndex,
                    'nodeIndex': nodeIndex_new,
                    'X_train': X_train,
                    'X_test': X_test,
                    'base_index': nodeIndex
                }
            )


class FSKernelTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, transform_expression, kernel, h, D,
                 category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.transform_expression = transform_expression
        self.kernel = kernel

    def require(self):
        return FSFeatureTransformTask(self.transform_expression,
                                      self.h, self.D, self.category,
                                      self.task_type)

    def __taskid__(self):
        return 'FSKernelTask_%s' % (str(
                                          containerHash(
                                                        list(
                                                             self.get_params().items()
                                                            )
                                                        )
                                           )
                                      )

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as inp:
            D = inp.query()

        graphIndex = D['graphIndex']
        X_train = D['X_train']
        X_test = D['X_test']
        train_l = X_train.shape[0]
        kernel = select_full(self.kernel)

        if kernel is None:
            raise ValueError('Unknown kernel %s' % self.kernel)

        X = vstack([X_train, X_test])

        X = normalize_gram(kernel(X))

        X_train = X[:train_l, :train_l]
        X_test = X[train_l:, :train_l]

        with self.output() as o:
            o.emit(
                {
                    'params': self.get_params(),
                    'graphIndex': graphIndex,
                    'data': X_train.tolist(),
                    'test': X_test.tolist()
                }
            )


class FSTuneLinearClassiferTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, ix, iy, Cs, h, D,
                 train_index, test_index,
                 category=None, task_type=None,
                 transform_expression='id'):
        self.ix = ix
        self.iy = iy
        self.Cs = Cs
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.transform_expression = transform_expression
        self.train_index = train_index
        self.test_index = test_index

    def require(self):
        return [BagLabelMatrixTask(self.h, self.D,
                                   self.category, self.task_type),
                FSFeatureTransformTask(self.transform_expression,
                                       self.h, self.D,
                                       self.train_index,
                                       self.test_index,
                                       self.category,
                                       self.task_type)]

    def __taskid__(self):
        return 'FSTuneLinearClassiferTask_%s' % (str(
                                          containerHash(
                                                        list(
                                                             self.get_params().items()
                                                            )
                                                        )
                                           )
                                      )

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as inp:
            label = inp.query()
        n = len(label['tools'])
        label = np.array(label['label_matrix'])[:, index(self.ix, self.iy, n)]

        with self.input()[1] as inp:
            D = inp.query()

        X = D['X_train']
        X_test = D['X_test']

        y = label[self.train_index]

        param_grid = {'C': self.Cs}
        out = {'param': self.get_params()}

        clf = GridSearchCV(
            LinearSVC(
                dual=X.shape[0] < X.shape[1],
            ),
            param_grid,
            scoring=make_scorer(f1_score),
            n_jobs=2,
            pre_dispatch=4,
            iid=False,
            cv=10
        )

        start_time = time.time()
        clf.fit(X, y)
        out['train_time'] = time.time() - start_time
        out['scores'] = clf.cv_results_['mean_test_score'].tolist()
        out['C'] = clf.best_params_

        start_time = time.time()
        prediction = clf.predict(X_test).tolist()
        out['prediction'] = prediction
        out['test_time'] = (time.time() - start_time)/X_test.shape[0]

        with self.output() as o:
            o.emit(out)


class FSEvaluateTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, tool_count, Cs, h, D, scores, train_index, test_index,
                 category=None, task_type=None,
                 transform_expression='id'):
        self.tool_count = tool_count
        self.Cs = Cs
        self.h = h
        self.D = D
        self.scores = scores
        self.train_index = train_index
        self.test_index = test_index
        self.category = category
        self.task_type = task_type
        self.transform_expression = transform_expression

    def require(self):
        out = [BagGraphIndexTask(self.h, self.D,
                                 self.category, self.task_type),
               BagFilterTask(self.h, self.D,
                             self.category, self.task_type),
               BagLabelMatrixTask(self.h, self.D,
                                  self.category, self.task_type)]
        for i in range(self.tool_count):
            out.append(
                FSTuneLinearClassiferTask(i, i, self.Cs, self.h, self.D,
                                          self.train_index, self.test_index,
                                          self.category, self.task_type,
                                          self.transform_expression)
            )

        for i in range(self.tool_count):
            for j in range(i+1, self.tool_count):
                out.append(
                    FSTuneLinearClassiferTask(i, j, self.Cs, self.h, self.D,
                                              self.train_index, self.test_index,
                                              self.category, self.task_type,
                                              self.transform_expression)
                )
        return out

    def __taskid__(self):
        return 'FSEvaluateTask_%s' % (str(
                                                      containerHash(
                                                                    list(
                                                                         self.get_params().items()
                                                                        )
                                                                    )
                                                       )
                                                  )

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def _build_maps(self):
        with self.input()[1] as i:
            D = i.query()
        map_to_labels = {k: v['label'] for k, v in D.items()}
        map_to_times = {k: v['time'] if 'time' in v else math.inf for k, v in D.items()}
        del D
        return map_to_labels, map_to_times

    def _build_score(self, labels, times):
        scores = {}
        for k in self.scores:
            scores[k] = select_score(k, labels, times)

        return scores

    @staticmethod
    def _index_map(index, mapping):
        mapping = {k: v for k, v in mapping.items() if k in index}
        V = [
           m for m in sorted(list(mapping.items()), key=lambda x: index[x[0]])
           ]
        graphs = [m[0] for m in V]
        return graphs, np.array([m[1] for m in V])

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        graphs = [g for g in sorted(
                   list(graphIndex.items()), key=lambda x: x[1]
                   )]

        graphs = [graphs[i][0] for i in self.test_index]

        with self.input()[2] as i:
            D = i.query()
        y = D['rankings']
        tools = D['tools']

        rank_expect = [y[i] for i in self.test_index]

        C_param = {}

        cols = []
        for i in range(3, len(self.input())):
            x, y = reverse_index(i - 3, self.tool_count)
            with self.input()[i] as i:
                D = i.query()
                col = np.array(D['prediction'])
                C_param[(x, y)] = D['C']
            cols.append(col)

        M = np.column_stack(cols)
        C_param = [(x, y, c) for (x, y), c in C_param.items()]

        rank_pred = [ranking(M[i, :], self.tool_count)
                     for i in range(M.shape[0])]

        for i in range(len(rank_pred)):
            rank_pred[i] = [tools[t] for t in rank_pred[i]]

        y, times = self._build_maps()
        scores = self._build_score(y, times)

        empirical = {}
        raw_empircal = {}
        for i, pred in enumerate(rank_pred):
            expected = rank_expect[i]
            g = graphs[i]
            for k, score in scores.items():
                if k not in empirical:
                    empirical[k] = 0.0
                    raw_empircal[k] = []
                s = score(pred, expected, g)
                empirical[k] += s / len(self.test_index)
                raw_empircal[k].append(s)

        with self.output() as emitter:
            emitter.emit(
               {
                   'parameter': self.get_params(),
                   'C': C_param,
                   'result': empirical,
                   'raw_results': raw_empircal
               }
               )
