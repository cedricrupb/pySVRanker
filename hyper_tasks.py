from .bag import ProgramBags, read_bag, normalize_gram, enumerateable, indexMap
from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService
from .bag_tasks import BagFilterTask, BagGraphIndexTask
from .pca_tasks import BagCalculateGramTask
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn import svm
import time
import math
from .rank_scores import select_score
from .bag_tasks import BagLabelMatrixTask, index, reverse_index, ranking


def divide(A, B):
    if A == 0 and B == 0:
        return 0
    return A / B


def accuracy(pred, expected):
    error_vector = pred - expected
    return 1 - np.count_nonzero(error_vector)/error_vector.shape[0]


def precision(pred, expected):
    p_index = pred.nonzero()
    err = pred[p_index] - expected[p_index]
    return 1 - divide(np.count_nonzero(err), np.count_nonzero(pred))


def recall(pred, expected):
    p_index = expected.nonzero()
    err = pred[p_index] - expected[p_index]
    return 1 - divide(np.count_nonzero(err), np.count_nonzero(expected))


def f1_score(pred, expected):
    p = precision(pred, expected)
    r = recall(pred, expected)
    return 2*divide((p*r), (p+r))


def cross_val(clf, X, y, metrics, cv=10):
    scores = {}
    loo = KFold(cv, shuffle=True)
    cross_index = list(range(X.shape[0]))
    for train_index, test_index in loo.split(cross_index):
        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]

        model = clone(clf)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        for name, metric in metrics.items():
            if name not in scores:
                scores[name] = []
            scores[name].append(metric(pred, y_test))

    return scores


class HyperCrossPredictTask(Task):
    out_dir = Parameter('./eval/')
    cv = Parameter(10)

    def __init__(self, ix, iy, C, h, D,
                 train_index, test_index,
                 eval=False,
                 category=None, task_type=None, kernel='linear'):
        self.ix = ix
        self.iy = iy
        self.C = C
        self.h = h
        self.D = D
        self.eval = eval
        self.train_index = train_index
        self.test_index = test_index
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        return [BagLabelMatrixTask(self.h, self.D,
                                   self.category, self.task_type),
                BagCalculateGramTask(self.h, self.D,
                                     category=self.category,
                                     task_type=self.task_type,
                                     kernel=self.kernel)]

    def __taskid__(self):
        return 'HyperCrossPredictTask_%s' % (str(
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
        with self.input()[0] as i:
            D = i.query()

        n = len(D['tools'])
        y = np.array(D['label_matrix'])[:, index(self.ix, self.iy, n)]
        del D

        with self.input()[1] as i:
            D = i.query()
        graphIndex = D['graphIndex']
        X = np.array(D['data'])
        del D

        out = {'param': self.get_params()}

        train_index = self.train_index
        test_index = self.test_index

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train = y[train_index]

        clf = svm.SVC(C=self.C, kernel='precomputed')

        if self.eval:
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1_score
            }

            start_time = time.time()
            scores = cross_val(clf, X_train, y_train, metrics, self.cv.value)
            eval_time = time.time() - start_time

            out['evaluation'] = scores
            out['eval_time'] = eval_time

        start_time = time.time()
        clf.fit(X_train, y_train)
        out['train_time'] = time.time() - start_time

        start_time = time.time()
        out['prediction'] = clf.predict(X_test).tolist()
        out['test_time'] = (time.time() - start_time)/X_test.shape[0]

        with self.output() as o:
            o.emit(out)


class HyperSingleOptimizationTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, ix, iy, CSet, h, D, score,
                 train_index, test_index,
                 category=None, task_type=None, kernel='linear'):
        self.ix = ix
        self.iy = iy
        self.CSet = CSet
        self.h = h
        self.D = D
        self.score = score
        self.train_index = train_index
        self.test_index = test_index
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        train, validate = train_test_split(self.train_index, test_size=0.33,
                                           random_state=0)
        out = [BagLabelMatrixTask(self.h, self.D,
                                  self.category, self.task_type),
               BagCalculateGramTask(self.h, self.D,
                                    category=self.category,
                                    task_type=self.task_type,
                                    kernel=self.kernel)]
        out.extend([
            HyperCrossPredictTask(
                self.ix, self.iy, c, self.h, self.D,
                train, validate, eval=True, category=self.category,
                task_type=self.task_type, kernel=self.kernel
            )
            for c in self.CSet
        ])
        return out

    def __taskid__(self):
        return 'HyperSingleOptimizationTask_%s' % (str(
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
        max_param = None
        max_score = -math.inf

        for ix in range(2, len(self.input())):
            inp = self.input()[ix]
            with inp as i:
                param = i.query()
            score = np.mean(param['evaluation'][self.score])
            if score > max_score:
                max_param = param
                max_score = score

        with self.input()[0] as i:
            D = i.query()

        n = len(D['tools'])
        y = np.array(D['label_matrix'])[:, index(self.ix, self.iy, n)]
        del D

        with self.input()[1] as i:
            D = i.query()
        graphIndex = D['graphIndex']
        X = np.array(D['data'])
        del D

        C = max_param['param']['C']

        out = {'param': self.get_params(), 'C': C}

        train_index = self.train_index
        test_index = self.test_index

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train = y[train_index]

        clf = svm.SVC(C=C, kernel='precomputed')

        start_time = time.time()
        clf.fit(X_train, y_train)
        out['train_time'] = time.time() - start_time

        start_time = time.time()
        out['prediction'] = clf.predict(X_test).tolist()
        out['test_time'] = (time.time() - start_time)/X_test.shape[0]

        with self.output() as o:
            o.emit(out)


class HyperSingleEvaluationTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, tool_count, CSet, h, D, sub_score,
                 scores,
                 train_index, test_index,
                 category=None, task_type=None, kernel='linear'):
        self.tool_count = tool_count
        self.CSet = CSet
        self.h = h
        self.D = D
        self.sub_score = sub_score
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
                HyperSingleOptimizationTask(
                    i, i, self.CSet, self.h, self.D,
                    self.sub_score, self.train_index, self.test_index,
                    self.category, self.task_type, self.kernel
                )
            )

        for i in range(self.tool_count):
            for j in range(i+1, self.tool_count):
                out.append(
                    HyperSingleOptimizationTask(
                        i, j, self.CSet, self.h, self.D,
                        self.sub_score, self.train_index, self.test_index,
                        self.category, self.task_type, self.kernel
                    )
                )
        return out

    def __taskid__(self):
        return 'HyperSingleEvaluationTask_%s' % (str(
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

        C_param = [[x, y, c] for (x, y), c in C_param.items()]
        M = np.column_stack(cols)

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


class CVHyperSingleEvalutionTask(Task):
        out_dir = Parameter('./eval/')

        def __init__(self, tool_count, Cs, h, D, sub_score,
                     scores, opt_score, full_index,
                     category=None, task_type=None, kernel='linear'):
            self.tool_count = tool_count
            self.Cs = Cs
            self.h = h
            self.D = D
            self.sub_score = sub_score
            self.scores = scores
            self.opt_score = opt_score
            self.full_index = full_index
            self.category = category
            self.task_type = task_type
            self.kernel = kernel

        def _index(self):
            if isinstance(self.full_index, int):
                return [x for x in range(self.full_index)]
            else:
                return self.full_index

        def require(self):
            index = np.array(self._index())
            loo = KFold(self.k.value, shuffle=True, random_state=0)
            return [
                HyperSingleEvaluationTask(
                    self.tool_count,
                    self.Cs,
                    self.h,
                    self.D,
                    self.sub_score,
                    self.scores,
                    train_index,
                    test_index,
                    self.category,
                    self.task_type,
                    self.kernel
                )
                for train_index, test_index in loo.split(index)
            ]

        def __taskid__(self):
            return 'CVHyperSingleEvalutionTask_%s' % (str(
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
