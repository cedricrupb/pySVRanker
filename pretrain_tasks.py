from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService
from .pca_tasks import BagCalculateGramTask
from .bag_tasks import BagLabelMatrixTask, BagFeatureTask, index, reverse_index
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from .bag_tasks import BagFilterTask, BagGraphIndexTask
from sklearn.model_selection import KFold
from .rank_scores import select_score
from sklearn.feature_extraction.text import TfidfTransformer


def ranking(row, n):
    N = np.zeros(n)

    for i in range(n):
        for j in range(i+1, n):
            better_i = row[index(i, j, n) - n] == 1
            N[i if better_i else j] += 1

    return N.argsort()[::-1]


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
        return [BagLabelMatrixTask(self.h, self.D,
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
        y = np.array(D['label_matrix'])[:, index(self.ix, self.iy, n)]
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

        graphs = [graphs[i][0] for i in self.test_index]

        with self.input()[2] as i:
            D = i.query()
        y = D['rankings']
        tools = D['tools']

        rank_expect = [y[i] for i in self.test_index]

        C_param = {}
        eval_param = {}

        cols = []
        cols_insample = []
        for i in range(3, len(self.input())):
            x, y = reverse_index(self.tool_count+(i - 3), self.tool_count)
            with self.input()[i] as i:
                D = i.query()
                col = np.array(D['prediction'])
                col_in = np.array(D['prediction_insample'])
            cols.append(col)
            cols_insample.append(col_in)

        M = np.column_stack(cols)
        M_in = np.column_stack(cols_insample)

        rank_pred = [ranking(M[i, :], self.tool_count)
                     for i in range(M.shape[0])]
        rank_pred_in = [ranking(M[i, :], self.tool_count)
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
                empirical[k] += s / len(self.test_index)
                if k not in empirical_in:
                    empirical_in[k] = 0.0
                s = score(pred_in, expected, g)
                empirical_in[k] += s / len(self.test_index)

        with self.output() as emitter:
            emitter.emit(
                {
                    'parameter': self.get_params(),
                    'result': empirical,
                    'result_insample': empirical_in
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
