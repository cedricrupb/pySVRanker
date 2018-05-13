import numpy as np
from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService
from .bag_tasks import BagLoadingTask, BagGraphIndexTask, BagNormalizeGramTask
from .bag_tasks import BagLabelMatrixTask, index, reverse_index, BagFilterTask
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
import time
import math
from .rank_scores import select_score
from sklearn.model_selection import KFold
from .bag import enumerateable
from scipy.spatial.distance import cdist
import os


def prob_ranking(row, n):
    N = np.zeros(n, dtype=np.float64)

    for i in range(n):
        p_i = row[index(i, i, n)]
        for j in range(n):
            if i < j:
                p_j = row[index(j, j, n)]
                faster_i = row[index(i, j, n)]
                N[i] += p_i * (1-p_j) + (p_i*p_j + (1-p_i)*(1-p_j))*faster_i
                N[j] += p_j * (1-p_i) + (p_i*p_j +
                                         (1-p_i)*(1-p_j))*(1-faster_i)

    return N.argsort()[::-1]


class BagCountTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type

    def require(self):
        return BagFilterTask(self.h, self.D,
                             self.category, self.task_type)

    def __taskid__(self):
        s = 'BagCountTask_%d_%d' % (self.h, self.D)
        if self.category is not None:
            s += '_'+str(containerHash(self.category))
        if self.task_type is not None:
            s += '_'+str(self.task_type)
        return s

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            D = i.query()

        count = {}
        for V in D.values():
            for n in V['kernel_bag'].keys():
                if n not in count:
                    count[n] = 0
                count[n] += 1

        with self.output() as o:
            o.emit(count)


class BagNodeIndexTask(Task):
    out_dir = Parameter('./gram/')
    max_features = Parameter(10000)

    def __init__(self, h, D, category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type

    def require(self):
        return [BagCountTask(h, self.D,
                             self.category, self.task_type)
                for h in range(self.h+1)]

    def __taskid__(self):
        s = 'BagNodeIndexTask_%d_%d' % (self.h, self.D)
        if self.category is not None:
            s += '_'+str(containerHash(self.category))
        if self.task_type is not None:
            s += '_'+str(self.task_type)
        return s

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        count = {}
        for inp in self.input():
            with inp as i:
                C = i.query()
            for n, c in C.items():
                if n in count:
                    count[n] += c
                else:
                    count[n] = c

        index = [
            x[0]
            for x in sorted(
                list(count.items()), key=lambda k: k[1], reverse=True
            )
        ]

        len = min(len(index), self.max_features.value)
        if len < 0:
            len = len(index)
        index = index[:len]
        index = {
            k: i for i, k in enumerate(index)
        }
        with self.output() as o:
            o.emit(
                index
            )


class PreparedFeatureTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, task_type=None, tfidf=True):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.tfidf = tfidf

    def require(self):
        out = [BagGraphIndexTask(self.h, self.D,
                                 self.category, self.task_type),
               BagNodeIndexTask(self.h, self.D,
                                self.category, self.task_type)]
        out.extend([BagFilterTask(h, self.D,
                                  self.category, self.task_type)
                    for h in range(self.h + 1)])
        return out

    def __taskid__(self):
        postfix = ''
        if self.tfidf:
            postfix += '_tfidf'
        s = 'PreparedFeatureTask_%d_%d' % (self.h, self.D)\
            + postfix
        if self.category is not None:
            s += '_'+str(containerHash(self.category))
        if self.task_type is not None:
            s += '_'+str(self.task_type)
        return s

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        with self.input()[1] as i:
            nodeIndex = i.query()

        row = []
        column = []
        data = []
        floatType = False

        row_shape = 0
        col_shape = 0

        for ix in range(2, len(self.input())):
            with self.input()[ix] as i:
                D = i.query()
            for ID, entry in D.items():
                if ID not in graphIndex:
                    continue
                gI = graphIndex[ID]
                row_shape = max(row_shape, gI)
                for n, c in entry['kernel_bag'].items():
                    if n not in nodeIndex:
                        continue
                    nI = nodeIndex[n]
                    col_shape = max(col_shape, nI)
                    floatType = floatType or isinstance(c, float)
                    row.append(gI)
                    column.append(nI)
                    data.append(c)

        dtype = np.float64 if floatType else np.uint64

        phi = coo_matrix((data, (row, column)),
                         shape=(row_shape+1,
                                col_shape+1),
                         dtype=dtype).tocsr()

        if self.tfidf:
            phi = TfidfTransformer(sublinear_tf=True).fit_transform(phi)

        NZ = phi.nonzero()
        data = phi[NZ].A
        shape = phi.get_shape()

        out = {
            'graphIndex': graphIndex,
            'nodeIndex': nodeIndex,
            'rows': NZ[0].tolist(),
            'columns': NZ[1].tolist(),
            'data': data.tolist()[0],
            'row_shape': shape[0],
            'column_shape': shape[1]
        }

        with self.output() as o:
            o.emit(out)


class PCAFeatureTask(Task):
    out_dir = Parameter('./gram/')
    components = Parameter(0.99)
    whiten = Parameter(False)

    def __init__(self, h, D, category=None, task_type=None,
                 kernel=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        if self.kernel is None:
            return PreparedFeatureTask(self.h, self.D,
                                       self.category, self.task_type)
        h = [h for h in range(self.h+1)]
        return BagNormalizeGramTask(h, self.D,
                                    self.category, self.task_type)

    def __taskid__(self):
        s = 'PCAFeatureTask_%d_%d' % (self.h, self.D)
        if self.category is not None:
            s += '_'+str(containerHash(self.category))
        if self.task_type is not None:
            s += '_'+str(self.task_type)
        if self.kernel is not None:
            s += '_'+str(self.kernel)
        return s

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            D = i.query()

        graphIndex = D['graphIndex']

        if self.kernel is None:
            X = coo_matrix((D['data'], (D['rows'], D['columns'])),
                           shape=(D['row_shape'],
                                  D['column_shape']),
                           dtype=np.float64).todense()
            pca = PCA(n_components=self.components.value,
                      whiten=self.whiten.value)
        else:
            X = np.array(D['data'])
            pca = KernelPCA(
                n_components=500,
                kernel='precomputed',
                n_jobs=-1,
                remove_zero_eig=True
            )

        X = pca.fit_transform(X)
        print('Reduced features: %s' % str(X.shape))

        with self.output() as o:
            o.emit(
                {
                    'graphIndex': graphIndex,
                    'matrix': X.tolist()
                }
            )


class SVDFeatureTask(Task):
    out_dir = Parameter('./gram/')
    components = Parameter(1000)

    def __init__(self, h, D, category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type

    def require(self):
        return PreparedFeatureTask(self.h, self.D,
                                   self.category, self.task_type)

    def __taskid__(self):
        s = 'SVDFeatureTask_%d_%d' % (self.h, self.D)
        if self.category is not None:
            s += '_'+str(containerHash(self.category))
        if self.task_type is not None:
            s += '_'+str(self.task_type)
        return s

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            D = i.query()

        graphIndex = D['graphIndex']

        X = coo_matrix((D['data'], (D['rows'], D['columns'])),
                       shape=(D['row_shape'],
                              D['column_shape']),
                       dtype=np.float64).tocsr()
        svd = TruncatedSVD(
            n_components=self.components.value
        )

        X = svd.fit_transform(X)
        print('Reduced features: %s' % str(X.shape))

        with self.output() as o:
            o.emit(
                {
                    'graphIndex': graphIndex,
                    'matrix': X.tolist()
                }
            )


class PCAKernelTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, kernel='linear', category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        return PCAFeatureTask(self.h, self.D, self.category, self.task_type)

    def __taskid__(self):
        s = 'PCAKernelTask_%d_%d_%s' % (self.h, self.D, self.kernel)
        if self.category is not None:
            s += '_'+str(containerHash(self.category))
        if self.task_type is not None:
            s += '_'+str(self.task_type)
        return s

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            D = i.query()
        graphIndex = D['graphIndex']
        X = np.matrix(D['matrix'])

        print(self.kernel)

        if self.kernel == 'linear':
            X = X.dot(X.transpose())
        else:
            X = cdist(X, X, self.kernel)
            X = np.ones(X.shape, dtype=np.float64) - X

        with self.output() as o:
            o.emit(
                {
                    'graphIndex': graphIndex,
                    'data': X.tolist()
                }
            )


class BagCalculateGramTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, task_type=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        if self.kernel.startswith('pca:'):
            return PCAKernelTask(
                self.h, self.D, self.kernel[4:], self.category, self.task_type
            )
        hSet = [h for h in range(self.h+1)]
        return BagNormalizeGramTask(hSet, self.D, self.category,
                                    self.task_type, self.kernel)

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = str(containerHash(self.category))

        tt = ''
        if self.task_type is not None:
            tt = '_'+str(self.task_type)

        return 'BagCalculateGramTask_%s_%d_%s_%s' % (str(
                                                      containerHash(self.h)
                                                      ),
                                                self.D, self.kernel, cat
                                                )\
            + tt

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            with self.output() as o:
                o.emit(i.query())


class TrainLRTask(Task):
    out_dir = Parameter('./eval/')
    cv = Parameter(10)
    max_iter = Parameter(100)

    def __init__(self, ix, iy, Cs, h, D, train_index, test_index,
                 category=None, task_type=None, kernel=None):
        self.ix = ix
        self.iy = iy
        self.Cs = Cs
        self.h = h
        self.D = D
        self.train_index = train_index
        self.test_index = test_index
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        out = [BagLabelMatrixTask(self.h, self.D,
                                  self.category, self.task_type),
               PCAFeatureTask(self.h, self.D,
               self.category, self.task_type, self.kernel)]

        if self.ix < self.iy:
            out.extend([
                TrainLRTask(
                    self.ix, self.ix, self.Cs, self.h, self.D,
                    self.train_index, self.test_index,
                    self.category, self.task_type, self.kernel
                ),
                TrainLRTask(
                    self.iy, self.iy, self.Cs, self.h, self.D,
                    self.train_index, self.test_index,
                    self.category, self.task_type, self.kernel
                )
            ])

        return out

    def __taskid__(self):
        return 'TrainLRTask_%s' % (str(
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

    def sample_weight(self):
        if self.ix < self.iy:
            with self.input()[2] as i:
                pred_i = np.array(i.query()['train_predict'])
            with self.input()[3] as i:
                pred_j = np.array(i.query()['train_predict'])

            out = np.zeros(pred_i.shape, dtype=np.float64)
            for i in range(out.shape[0]):
                out[i] = pred_i[i] * pred_j[i] + (1 - pred_i[i])*(1 - pred_j[i])

            return out

    def run(self):
        with self.input()[0] as i:
            D = i.query()

        n = len(D['tools'])
        y = np.array(D['label_matrix'])[:, index(self.ix, self.iy, n)]
        del D

        with self.input()[1] as i:
            D = i.query()

        graphIndex = D['graphIndex']
        X = np.matrix(D['matrix'])
        del D

        out = {'param': self.get_params()}

        train_index = self.train_index
        test_index = self.test_index

        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]

        clf = LogisticRegressionCV(
            Cs=self.Cs,
            cv=self.cv.value,
            dual=False,
            max_iter=self.max_iter.value,
            solver='saga',
            scoring=make_scorer(f1_score),
            n_jobs=-1
        )

        start_time = time.time()
        clf.fit(X_train, y_train, sample_weight=self.sample_weight())

        if self.ix == self.iy:
            prediction = clf.predict_proba(X_train)
            prediction = [x for x in
                          (dict(zip(clf.classes_, x))for x in prediction)]
            out['train_predict'] = [P[1.0] for P in prediction]

        out['train_time'] = time.time() - start_time
        out['scores'] = {k: v.tolist() for k, v in clf.scores_.items()}
        out['C'] = dict(zip(clf.classes_, clf.C_))

        start_time = time.time()
        prediction = clf.predict_proba(X_test)
        prediction = [x for x in
                      (dict(zip(clf.classes_, x))for x in prediction)]
        prediction = [P[1.0] for P in prediction]
        out['prediction'] = prediction
        out['test_time'] = (time.time() - start_time)/X_test.shape[0]

        with self.output() as o:
            o.emit(out)


class EvaluateLRTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, tool_count, Cs, h, D, scores, train_index, test_index,
                 category=None, task_type=None, kernel=None):
        self.tool_count = tool_count
        self.Cs = Cs
        self.h = h
        self.D = D
        self.scores = scores
        self.train_index = train_index
        self.test_index = test_index
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        out = [BagGraphIndexTask(self.h, self.D,
                                 self.category, self.task_type),
               BagFilterTask(self.h, self.D,
                             self.category, self.task_type),
               BagLabelMatrixTask(self.h, self.D,
                                  self.category, self.task_type)]
        for i in range(self.tool_count):
            out.append(
                TrainLRTask(i, i, self.Cs, self.h, self.D,
                            self.train_index, self.test_index,
                            self.category, self.task_type, self.kernel)
            )

        for i in range(self.tool_count):
            for j in range(i+1, self.tool_count):
                out.append(
                    TrainLRTask(i, j, self.Cs, self.h, self.D,
                                self.train_index, self.test_index,
                                self.category, self.task_type, self.kernel)
                )
        return out

    def __taskid__(self):
        return 'EvaluateLRTask_%s' % (str(
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

        rank_pred = [prob_ranking(M[i, :], self.tool_count)
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


class CVEvaluateLRTask(Task):
        out_dir = Parameter('./eval/')
        k = Parameter(10)

        def __init__(self, tool_count, Cs, h, D, scores,
                     opt_score, full_index,
                     category=None, task_type=None):
            self.tool_count = tool_count
            self.Cs = Cs
            self.h = h
            self.D = D
            self.scores = scores
            self.opt_score = opt_score
            self.full_index = full_index
            self.category = category
            self.task_type = task_type

        def _index(self):
            if isinstance(self.full_index, int):
                return [x for x in range(self.full_index)]
            else:
                return self.full_index

        def require(self):
            index = np.array(self._index())
            loo = KFold(self.k.value, shuffle=True, random_state=0)
            return [
                EvaluateLRTask(
                    self.tool_count,
                    self.Cs,
                    self.h,
                    self.D,
                    self.scores,
                    train_index.tolist(),
                    test_index.tolist(),
                    self.category,
                    self.task_type
                )
                for train_index, test_index in loo.split(index)
            ]

        def __taskid__(self):
            return 'CVEvaluateLRTask_%s' % (str(
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
            out = []
            for inp in self.input():
                D = {}
                with inp as i:
                    T = i.query()
                D['C'] = T['C']
                D['result'] = T['result']
                del T
                out.append(D)
            max_C = max(out, key=lambda D: D['result'][self.opt_score])['C']

            results = {}
            for i, D in enumerate(out):
                for k, f in D['result'].items():
                    if k not in results:
                        results[k] = np.zeros(len(out), dtype=np.float64)
                    results[k][i] = f

            for k in results.keys():
                results[k] = (results[k].mean(), results[k].std())

            with self.output() as o:
                o.emit(
                    {
                        'param': self.get_params(),
                        'C': max_C,
                        'results': results
                    }
                )
