from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget, FileTarget
from pyTasks.target import JsonService
from .pca_tasks import BagCalculateGramTask
from .bag_tasks import BagLabelMatrixTask, BagFeatureTask, index, reverse_index
from .czech_tasks import accumulate_label
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from .bag_tasks import BagFilterTask, BagGraphIndexTask
from sklearn.model_selection import KFold
from .rank_scores import select_score
from sklearn.feature_extraction.text import TfidfTransformer
import math
from os.path import abspath, join, isdir, isfile
import subprocess
from subprocess import PIPE
import re
from sklearn.base import clone


def partial_ranking(row, n):
    N = np.zeros(n)

    for i in range(n):
        for j in range(i+1, n):
            if row[index(i, j, n) - n] == 0:
                continue
            better_i = row[index(i, j, n) - n] == 1
            N[i if better_i else j] += 1

    return N.argsort()[::-1]


def ranking(row, n):
    N = np.zeros(n)

    for i in range(n):
        for j in range(i+1, n):
            better_i = row[index(i, j, n) - n] == 1
            N[i if better_i else j] += 1

    return N.argsort()[::-1]


def d_accumulate_label(ii, jj, ij):
    y = np.zeros(ii.shape)

    for i in range(len(ii)):
        y[i] = ii[i]*(1 - jj[i]) + (ii[i]*jj[i] + (1-ii[i])*(1-jj[i])) * ij[i]

    return y


def is_valid(X):
    return X['time'] < 900


def is_better(X, Y):
    return X['solve'] > Y['solve'] or\
            (X['solve'] == Y['solve'] and X['time'] < Y['time'])


class SVCompLabelMatrixTask(Task):
    out_dir = Parameter('./gram/')
    allowed = Parameter(None)

    def __init__(self, h, D, category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type

    def require(self):
        return [BagGraphIndexTask(self.h,
                                  self.D,
                                  self.category, self.task_type),
                BagFilterTask(self.h, self.D,
                              self.category, self.task_type)]

    def __taskid__(self):
        s = 'SVCompLabelMatrixTask_%d_%d' % (self.h, self.D)
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

    @staticmethod
    def common_tools(bag):
        F = len(bag)
        C = {}
        for B in bag.values():
            for tool in B['label'].keys():
                if tool not in C:
                    C[tool] = 0
                C[tool] += 1
        tools = [k for k, v in C.items()]  # if v == F]

        for B in bag.values():
            for d in [d for d in B['label'].keys() if d not in tools]:
                del B['label'][d]

        return tools

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        with self.input()[1] as i:
            bag = i.query()

        self.tools = BagLabelMatrixTask.common_tools(bag)

        if self.allowed.value is not None:
            self.tools = [t for t in self.tools if t in self.allowed.values]

        n = len(self.tools)
        label_matrix = np.zeros((graphIndex['counter'], int(0.5*n*(n+1))))
        rankings = np.array([None]*graphIndex['counter'])
        tool_index = {t: i for i, t in enumerate(self.tools)}

        for k, B in bag.items():
            if k not in graphIndex:
                continue
            index_g = graphIndex[k]
            for tool_x in self.tools:
                index_x = tool_index[tool_x]
                label_x = None

                if tool_x in B['label'] and is_valid(B['label'][tool_x]):
                    label_x = B['label'][tool_x]

                for tool_y in self.tools:
                    index_y = tool_index[tool_y]

                    label_y = None

                    if tool_y in B['label'] and is_valid(B['label'][tool_y]):
                        label_y = B['label'][tool_y]

                    if index_y > index_x:
                        val = 0

                        if label_x is None:
                            if label_y is not None:
                                val = -1
                        else:
                            if label_y is None:
                                val = 1
                            else:
                                val = 1 if is_better(label_x, label_y) else -1

                        label_matrix[index_g, index(index_x, index_y, n) - n] = val

            rank = partial_ranking(label_matrix[index_g, :], n)
            rankings[index_g] = [self.tools[i] for i in rank]

        with self.output() as o:
            o.emit(
                {
                    'tools': self.tools,
                    'label_matrix': label_matrix.tolist(),
                    'rankings': rankings.tolist()
                }
            )


class SVGraphTask(Task):
    out_dir = Parameter("./dfs")
    cpaChecker = Parameter("")
    bench_path = Parameter("")
    localize = Parameter(None)

    def init(self, task):
        self.task = task

    def _localize(self, path):
        if self.localize.value is not None:
            return path.replace(self.localize.value[0],
                                self.localize.value[1])
        return path

    def require(self):
        return None

    def __taskid__(self):
        tid = self.task
        tid = tid.replace("/", "_")
        tid = tid.replace("\\.", "_")
        return tid

    def output(self):
        return FileTarget(
            "%s/%s.dfs" % (self.out_dir.value, self.__taskid__())
        )

    def run(self):
        out_path = self.output().path

        path_to_source = self._localize(abspath("%s/%s" % (self.bench_path.value, self.task)))

        __path_to_cpachecker__ = self.cpaChecker.value
        cpash_path = join(__path_to_cpachecker__, 'scripts', 'cpa.sh')

        if not isdir(__path_to_cpachecker__):
            raise ValueError('CPAChecker directory not found')
        if not (isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
            raise ValueError('path_to_source is no valid filepath. [%s]' % path_to_source)

        proc = subprocess.run([cpash_path,
                               '-graphgen',
                               '-heap', self.heap.value,
                               path_to_source,
                               '-setprop', "graphGen.output=%s" % out_path
                               ],
                              check=False, stdout=PIPE, stderr=PIPE)
        match_vresult = re.search(r'Verification\sresult:\s([A-Z]+)\.', str(proc.stdout))
        if match_vresult is None:
            raise ValueError('Invalid output of CPAChecker.')


class KernelSVMTask(Task):
    out_dir = Parameter('./svm/')
    cv = Parameter(10)

    def __init__(self, ix, iy, C, h, D,
                 train_index, test_index,
                 category=None, task_type=None,
                 kernel='linear'):
        self.ix = ix
        self.iy = iy
        self.C = C
        self.h = h
        self.D = D
        self.test_index = test_index
        self.train_index = train_index
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
        return 'KernelSVMTask_%s' % (str(
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

        tools = D['tools']
        n = len(D['tools'])
        y =\
            accumulate_label(
                np.array(D['label_matrix'])[:, index(self.ix, self.ix, n)],
                np.array(D['label_matrix'])[:, index(self.iy, self.iy, n)],
                np.array(D['label_matrix'])[:, index(self.ix, self.iy, n)]
            )
        del D

        with self.input()[1] as i:
            D = i.query()
        graphIndex = D['graphIndex']
        X = np.array(D['data'])
        del D

        rev = [k for k, v in sorted(list(graphIndex.items()), key=lambda x: x[1])]
        out = {'param': self.get_params()}

        train_index = self.train_index
        test_index = self.test_index

        X_train = X[train_index][:, train_index]
        X_test = X[test_index][:, train_index]
        y_train = y[train_index]

        X = np.nan_to_num(X)

        if np.isnan(X).any():
            raise Exception("Fail for X")

        if np.isnan(y).any():
            raise Exception("Fail for y in %d, %d" % (self.ix, self.iy))

        clf = SVC(kernel='precomputed')
        params = {
            'C': self.C
        }
        scores = ['f1', 'precision', 'recall', 'accuracy']
        clf = GridSearchCV(clf, params, scores, cv=self.cv.value, refit='f1')
        clf.fit(X, y)
        svc = SVC(kernel='precomputed', C=clf.best_params_['C'])
        svc.fit(X, y)

        out['C'] = clf.best_params_['C']
        out['result'] = clf.cv_results_
        del out['result']['params']
        for k in out['result'].keys():
            out['result'][k] = out['result'][k].tolist()
        out['coef'] = {}

        coef = svc.dual_coef_
        for i in range(coef.shape[1]):
            ix = svc.support_[i]
            jx = ix
            out['coef'][rev[jx]] = coef[0, i]

        out['y'] = {}
        for i in range(y_train.shape[0]):
            if i in svc.support_:
                j = i
                _y = y[i]
                out['y'][rev[j]] = 1 if _y > 0 else -1

        out['intercept'] = svc.intercept_[0]
        out['prediction'] = clf.predict(X).tolist()
        out['prediction_insample'] = svc.predict(X).tolist()

        all_predict = svc.predict(X).tolist()
        all_predict = [tools[self.ix if p == 1 else self.iy] for p in all_predict]
        out['prediction_all'] = all_predict

        with self.output() as o:
            o.emit(out)


class LinearSVMTask(Task):
    out_dir = Parameter('./svm/')
    cv = Parameter(10)

    def __init__(self, ix, iy, C, h, D,
                 train_index, test_index,
                 category=None, task_type=None):
        self.ix = ix
        self.iy = iy
        self.C = C
        self.h = h
        self.D = D
        self.test_index = test_index
        self.train_index = train_index
        self.category = category
        self.task_type = task_type

    def require(self):
        return [SVCompLabelMatrixTask(self.h, self.D,
                                      self.category, self.task_type),
                BagFeatureTask(self.h, self.D,
                               category=self.category,
                               task_type=self.task_type)]

    def __taskid__(self):
        return 'LinearSVMTask_%s' % (str(
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
        y = np.array(D['label_matrix'])[:, index(self.ix, self.iy, n) - n]
        del D

        with self.input()[1] as i:
            D = i.query()
        graphIndex = D['graphIndex']
        nodeIndex = D['nodeIndex']
        X = D['features']
        del D

        rev = [x for x, c in sorted(list(nodeIndex.items()), key=lambda X: X[1])]

        out = {'param': self.get_params()}

        train_index = self.train_index
        test_index = self.test_index

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]

        clf = LinearSVC(dual=False)
        params = {
            'C': self.C
        }
        scores = ['f1', 'precision', 'recall', 'accuracy']
        clf = GridSearchCV(clf, params, scores, cv=self.cv.value, refit='f1')
        clf.fit(X_train, y_train)
        svc = LinearSVC(C=clf.best_params_['C'], dual=False)
        svc.fit(X, y)

        out['C'] = clf.best_params_['C']
        out['result'] = clf.cv_results_
        del out['result']['params']
        for k in out['result'].keys():
            out['result'][k] = out['result'][k].tolist()
        out['coef'] = {}

        coef = svc.coef_
        for i in range(coef.shape[1]):
            out['coef'][rev[i]] = coef[0, i]

        out['intercept'] = svc.intercept_[0]
        out['prediction'] = clf.predict(X_test).tolist()
        out['prediction_insample'] = svc.predict(X_test).tolist()

        with self.output() as o:
            o.emit(out)


class SVMSingleEvaluationTask(Task):
    out_dir = Parameter('./svm/')

    def __init__(self, tool_count, CSet, h, D,
                 scores,
                 train_index, test_index,
                 category=None, task_type=None, kernel='linear'):
        self.tool_count = tool_count
        self.CSet = CSet
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
            for j in range(i+1, self.tool_count):
                if self.kernel == 'linear':
                    out.append(
                        LinearSVMTask(
                            i, j, self.CSet, self.h, self.D,
                            self.train_index, self.test_index,
                            self.category, self.task_type
                        )
                    )
                else:
                    out.append(
                        KernelSVMTask(
                            i, j, self.CSet, self.h, self.D,
                            self.train_index, self.test_index,
                            self.category, self.task_type, self.kernel
                        )
                    )
        return out

    def __taskid__(self):
        return 'SVMSingleEvaluationTask_%s' % (str(
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

        with self.input()[2] as i:
            D = i.query()
        y = D['rankings']
        tools = D['tools']

        graphs = [graphs[i][0] for i in range(len(y))]

        rank_expect = y

        svm_param = {}

        cols = []
        cols_insample = []
        for i in range(3, len(self.input())):
            x, y = reverse_index(self.tool_count+(i - 3), self.tool_count)
            with self.input()[i] as i:
                D = i.query()
                col = np.array(D['prediction'])
                col_in = np.array(D['prediction_insample'])
                xI = tools[x]
                yI = tools[y]
                if xI not in svm_param:
                    svm_param[xI] = {}
                if yI not in svm_param[xI]:
                    svm_param[xI][yI] = {}
                svm_param[xI][yI]['coef'] = D['coef']
                svm_param[xI][yI]['intercept'] = D['intercept']
                if 'y' in D:
                    svm_param[xI][yI]['y'] = D['y']
            cols.append(col)
            cols_insample.append(col_in)

        M = np.column_stack(cols)
        M_in = np.column_stack(cols_insample)

        rank_pred = [partial_ranking(M[i, :], self.tool_count)
                     for i in range(M.shape[0])]
        rank_pred_in = [partial_ranking(M[i, :], self.tool_count)
                        for i in range(M_in.shape[0])]

        for i in range(len(rank_pred)):
            rank_pred[i] = [tools[t] for t in rank_pred[i]]
        for i in range(len(rank_pred_in)):
            rank_pred_in[i] = [tools[t] for t in rank_pred_in[i]]

        y, times = self._build_maps()
        scores = self._build_score(y, times)

        empirical = {}
        empirical_in = {}
        for i, pred in enumerate(rank_pred):
            pred_in = rank_pred_in[i]
            print(pred_in)
            expected = rank_expect[i]
            g = graphs[i]
            for k, score in scores.items():
                if k not in empirical:
                    empirical[k] = 0.0
                s = score(pred, expected, g)
                empirical[k] += s / len(y)
                if k not in empirical_in:
                    empirical_in[k] = 0.0
                s = score(pred_in, expected, g)
                empirical_in[k] += s / len(y)

        with self.output() as emitter:
            emitter.emit(
                {
                    'parameter': self.get_params(),
                    'result': empirical,
                    'result_insample': empirical_in,
                    'svm_param': svm_param
                }
            )


class CVSVMSingleEvalutionTask(Task):
        out_dir = Parameter('./svm/')
        k = Parameter(10)

        def __init__(self, tool_count, Cs, h, D,
                     scores, opt_score, full_index,
                     category=None, task_type=None, kernel='linear'):
            self.tool_count = tool_count
            self.Cs = Cs
            self.h = h
            self.D = D
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
                SVMSingleEvaluationTask(
                    self.tool_count,
                    self.Cs,
                    self.h,
                    self.D,
                    self.scores,
                    train_index.tolist(),
                    test_index.tolist(),
                    self.category,
                    self.task_type,
                    self.kernel
                )
                for train_index, test_index in loo.split(index)
            ]

        def __taskid__(self):
            return 'CVSVMSingleEvalutionTask_%s' % (str(
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
                D['result'] = T['result']
                D['result_insample'] = T['result_insample']
                del T
                out.append(D)

            results = {}
            for i, D in enumerate(out):
                for k, f in D['result'].items():
                    if k not in results:
                        results[k] = np.zeros(len(out), dtype=np.float64)
                    results[k][i] = f

            for k in results.keys():
                results[k] = (results[k].mean(), results[k].std())

            results_in = {}
            for i, D in enumerate(out):
                for k, f in D['result_insample'].items():
                    if k not in results_in:
                        results_in[k] = np.zeros(len(out), dtype=np.float64)
                    results_in[k][i] = f

            for k in results_in.keys():
                results_in[k] = (results_in[k].mean(), results_in[k].std())

            with self.output() as o:
                o.emit(
                    {
                        'param': self.get_params(),
                        'results': results,
                        'results_insample': results_in
                    }
                )


def cross_predict(clf, X, y, cv=10):
    scores = {}
    loo = KFold(cv, shuffle=True)
    cross_index = list(range(X.shape[0]))
    prediction = np.zeros(y.shape)
    for train_index, test_index in loo.split(cross_index):
        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]

        model = clone(clf)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        prediction[test_index] = pred

    return prediction


class KernelSVMTimeTask(Task):
    out_dir = Parameter('./svm/')
    cv = Parameter(10)

    def __init__(self, ix, iy, C, h, D,
                 category=None, task_type=None,
                 kernel='linear'):
        self.ix = ix
        self.iy = iy
        self.C = C
        self.h = h
        self.D = D
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
        return 'KernelSVMTimeTask_%s' % (str(
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

        tools = D['tools']
        n = len(D['tools'])
        y =\
            accumulate_label(
                np.array(D['label_matrix'])[:, index(self.ix, self.ix, n)],
                np.array(D['label_matrix'])[:, index(self.iy, self.iy, n)],
                np.array(D['label_matrix'])[:, index(self.ix, self.iy, n)]
            )
        del D

        with self.input()[1] as i:
            D = i.query()
        graphIndex = D['graphIndex']
        X = np.array(D['data'])
        del D

        rev = [k for k, v in sorted(list(graphIndex.items()), key=lambda x: x[1])]
        out = {'param': self.get_params()}

        clf = SVC(C=self.C, kernel='precomputed')

        all_predict = cross_predict(clf, X, y, cv=self.cv.value).tolist()
        all_predict = [tools[self.ix if p == 1 else self.iy] for p in all_predict]
        out['prediction'] = all_predict

        with self.output() as o:
            o.emit(out)
