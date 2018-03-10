from pyTasks import task
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from pyTasks.task import Task, Parameter, Optional
from pyTasks.utils import containerHash
from ranking_task import DefineClassTask
from gram_tasks import NormalizedWLKernelTask, parseList, MDSTask, ExtractKernelBagTask
from pyTasks.target import CachedTarget, LocalTarget, JsonService
from sklearn.model_selection import KFold
import random
import math
from sklearn.grid_search import ParameterGrid
from prepare_tasks import GraphIndexTask
from pyTasks.utils import tick
import networkx as nx
from tqdm import tqdm
import json

__unknownClass = 'UNKNOWN'
__pos = 'correct'


def label(y):
    y_out = [__unknownClass] * len(y)

    for i, _y in enumerate(y):
        if y[0] is not y[1]:
            if y[0] is 'correct':
                y_out[i] = 'Tester'
            else:
                y_out[i] = 'Verifier'
        else:
            y_out[i] = y[2]
    return y_out


def evalTime(t_rank):
    testSet = set([])
    allSet = set([])

    for u, smaller, v in t_rank:
        if smaller:
            testSet.add(v)
        allSet.add(u)
        allSet.add(v)

    out = []
    for k in allSet:
        if k not in testSet:
            out.append(k)

    if len(out) > 1:
        return 'UNKNOWN'

    return out[0]


class MajorityOrSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self._classifier = None
        for _y in y:
            if self._classifier is None:
                self._classifier = _y
            elif self._classifier != _y:
                self._classifier = SVC(C=self.C, kernel='precomputed')
                break

        if isinstance(self._classifier, SVC):
            self._classifier.fit(X, y)
        else:
            print('!!!! Majority %s !!!' % str(self._classifier))

    def predict(self, X):
        if isinstance(self._classifier, SVC):
            return self._classifier.predict(X)
        else:
            return [self._classifier] * len(X)

    def score(self, X, y):
        score = np.zeros(len(y))

        y_pred = self.predict(X)
        for i, _y in enumerate(y):
            if y_pred[i] == _y:
                score[i] = 1

        return score.mean()


class TimeScoreSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=(1.0, 1.0, 1.0)):
        self.C = C

    def fit(self, X, y):
        self.__tester = MajorityOrSVC(C=self.C[0])
        self.__verifier = MajorityOrSVC(C=self.C[1])
        self.__time = MajorityOrSVC(C=self.C[2])

        y_tester = [y_[0] for y_ in y]
        y_verifier = [y_[1] for y_ in y]
        y_time = [y_[2] for y_ in y]
        self.__tester.fit(X, y_tester)
        self.__verifier.fit(X, y_verifier)
        self.__time.fit(X, y_time)

    def predict_tripel(self, X):
        y_tester = self.__tester.predict(X)
        y_verifier = self.__verifier.predict(X)
        y_time = self.__time.predict(X)

        return [
            (y_tester[i], y_verifier[i], y_time[i]) for i in range(len(y_tester))
        ]

    def predict(self, X):

        y_tripel = self.predict_tripel(X)
        return label(y_tripel)

    def score(self, X, y):
        if len(X) != len(y):
            raise ValueError(
                    'X and y has to have the same dimension: %d and %d'
                    % (len(X), len(y)))

        p = self.predict(X)
        score = np.array([
            1 if p[i] == y[i] else 0
            for i in range(len(y))
        ])

        return score.mean()


class CGridTask(Task):
    out_dir = Parameter('./evaluation/')
    C_tester = Parameter([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    C_verifier = Parameter([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    C_time = Parameter([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    timeout = Parameter(None)

    def __init__(self, graphs, train_index, h, D):
        self.graphs = graphs
        self.train_index = train_index
        self.h = h
        self.D = D

    def require(self):
        return [
            NormalizedWLKernelTask(self.graphs, self.h, self.D),
            DefineClassTask(self.graphs)
            ]

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return "CGrid_%d_%d_%s_%s" %\
                    (self.h, self.D, str(containerHash(self.train_index)),
                     str(containerHash(self.graphs)))

    def __repr__(self):
        return 'CGrid(h: %d, D: %d)' % (self.h, self.D)

    @staticmethod
    def _k_fold_cv_gram(gram_matrix, y, C, folds=10, shuffle=True):
        """
        K-fold cross-validation using a precomputed gram matrix
        """
        scores = []
        loo = KFold(folds, shuffle=shuffle, random_state=random.randint(0, 100))
        for train_index, test_index in loo.split(np.arange(len(y))):

            X_train, X_test = gram_matrix[train_index][:, train_index], gram_matrix[test_index][:, train_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = TimeScoreSVC(C=C)
            clf.fit(X_train, y_train)

            score = clf.score(X_test, label(y_test))
            scores.append(score)
        return np.mean(scores), np.std(scores)

    def run(self):
        with self.input()[0] as i:
            graphIndex, GR = i.query()

        with self.input()[1] as i:
            R = i.query()

        train_index = np.array(self.train_index)
        test_index = list(range(len(GR)))
        test_index = [i for i in test_index if i not in train_index]

        y_bin = [None] * len(GR)
        for index, D in R.items():
            if index not in graphIndex:
                continue
            gI = graphIndex[index]
            score = D['score']
            time = evalTime(D['time_rank'])
            y_bin[gI] = (score['IUV'], score['ESBMC'], time)
        y_bin = np.array(y_bin)
        y_test = y_bin[test_index]
        y_train = y_bin[train_index]

        TGR = GR[train_index][:, train_index]
        EGR = GR[test_index][:, train_index]

        param_grid = {'C_tester': self.C_tester.value,
                      'C_verifier': self.C_verifier.value,
                      'C_time': self.C_time.value}
        max_mean = -math.inf
        max_param = None
        for params in ParameterGrid(param_grid):

            tick(self)

            mean, _ = self._k_fold_cv_gram(TGR, y_train, (params['C_tester'],
                                           params['C_verifier'],
                                           params['C_time']))
            if mean > max_mean:
                max_mean = mean
                max_param = params

        max_param['mean'] = max_mean
        max_param['h'] = self.h
        max_param['D'] = self.D
        max_param['train_matrix'] = TGR.tolist()
        max_param['train_y'] = y_train.tolist()
        max_param['test_matrix'] = EGR.tolist()
        max_param['test_y'] = y_test.tolist()

        with self.output() as o:
            o.emit(max_param)


class hDGridTask(Task):
    out_dir = Parameter('./evaluation/')
    timeout = Parameter(None)

    def __init__(self, graphs, train_index, h_Set, D_Set):
        self.graphs = graphs
        self.train_index = train_index
        self.h_Set = h_Set
        self.D_Set = D_Set

    def require(self):
        param_grid = {'h': self.h_Set, 'D': self.D_Set}
        out = []
        for params in ParameterGrid(param_grid):
            out.append(
                        Optional(
                            CGridTask(self.graphs,
                                      self.train_index,
                                      params['h'],
                                      params['D']
                                      )
                        )
                    )
        return out

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return "hDGrid_%s_%s_%s_%s" %\
                    (str(containerHash(self.graphs)),
                     str(containerHash(self.train_index)),
                     str(containerHash(self.h_Set)),
                     str(containerHash(self.D_Set)))

    def __repr__(self):
        return 'hDGrid(h: %d, D: %d)' % (self.h, self.D)

    def run(self):
        max_param = None
        max_mean = -math.inf

        for inp in self.input():

            tick(self)

            if inp is not None:
                with inp as i:
                    param = i.query()
                    if param['mean'] > max_mean:
                        max_mean = param['mean']
                        max_param = param
                    del param

        with self.output() as o:
            o.emit(max_param)


class EvaluationTask(Task):
    out_dir = Parameter('./evaluation/')
    folds = Parameter(10)

    def __init__(self, graphs, h_Set, D_Set):
        self.graphs = graphs
        self.h_Set = h_Set
        self.D_Set = D_Set

    def require(self):
        loo = KFold(self.folds.value,
                    shuffle=True,
                    random_state=random.randint(0, 100))
        out = []
        for train_index, _ in loo.split(np.arange(len(self.graphs))):
            out.append(
                    Optional(
                        hDGridTask(
                            self.graphs,
                            train_index.tolist(),
                            self.h_Set,
                            self.D_Set
                        )
                    )
            )
        return out

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return "Evaluation_%s_%s_%s" %\
                    (str(containerHash(self.graphs)),
                     str(containerHash(self.h_Set)),
                     str(containerHash(self.D_Set)))

    def __repr__(self):
        return 'Evaluation(h: %s, D: %s)' % (str(self.h_Set), str(self.D_Set))

    def run(self):
        scores = []

        for index, inp in enumerate(self.input()):
            if inp is not None:
                with inp as i:
                    param = i.query()
                clf = TimeScoreSVC(C=(param['C_tester'], param['C_verifier'],
                                   param['C_time']))
                train_matrix = np.array(param['train_matrix'])
                y_train = np.array(param['train_y'])

                del param['train_matrix']
                del param['train_y']

                clf.fit(train_matrix, y_train)

                test_matrix = np.array(param['test_matrix'])
                y_test = np.array(param['test_y'])

                del param['test_matrix']
                del param['test_y']

                score = clf.score(test_matrix, label(y_test))

                param['mean'] = score
                scores.append(param)

        scores = sorted(
            scores,
            key=lambda x: x['mean'],
            reverse=True
        )

        stat = {}
        stat['max'] = scores[0]
        stat['mean'] = {}

        for k in scores[0]:
            li = np.array([sc[k] for sc in scores])
            stat['mean'][k] = (li.mean(), li.std())

        with self.output() as o:
            o.emit(stat)


class EvaluationAndSettingTask(Task):
    out_dir = Parameter('./evaluation/')

    def __init__(self, graphs, h_Set, D_Set):
        self.graphs = graphs
        self.h_Set = h_Set
        self.D_Set = D_Set

    def require(self):
        return [EvaluationTask(
                                self.graphs,
                                self.h_Set,
                                self.D_Set),
                GraphIndexTask()]

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return "EvaluationAndSetting_%s_%s_%s" %\
                    (str(containerHash(self.graphs)),
                     str(containerHash(self.h_Set)),
                     str(containerHash(self.D_Set)))

    def __repr__(self):
        return 'EvaluationAndSetting(h: %d, D: %d)' % (self.h, self.D)

    def run(self):
        with self.input()[0] as i:
            param = i.query()

        with self.input()[1] as i:
            index = i.query()

        categories = []
        for c, D in index['categories'].items():
            for g in self.graphs:
                if g in D:
                    categories.append(c)
                    break

        D = {}
        D['categories'] = categories
        D['h_Set'] = self.h_Set
        D['D_Set'] = self.D_Set

        param['setting'] = D

        with self.output() as o:
            o.emit(param)


if __name__ == '__main__':
    from classification_tasks import EvaluationAndSettingTask
    from prepare_tasks import GraphIndexTask
    from gram_tasks import MDSTask

    config = {
        "GraphSubTask": {
            "graphPath": "/Users/cedricrichter/Documents/Arbeit/Ranking/PyPRSVT/static/results-tb-raw/",
            "graphOut": "./test/",
            "cpaChecker": "/Users/cedricrichter/Documents/Arbeit/Ranking/cpachecker"
                },
        "GraphConvertTask": {
            "graphOut": "./test/"
        },
        "CategoryLookupTask": {
            "graphPaths": "/home/cedricr/predicate/PyPRSVT/static/results-tb-raw/"
        },
        "MemcachedTarget": {
            "baseDir": "./cache/"
        },
        "GraphIndexTask": {
            "categories": ["array-examples", "loops", "reducercommutativity"]
        },
        "GraphPruningTask": {
            "graphOut": "./test/"
        }
            }

    injector = task.ParameterInjector(config)
    planner = task.TaskPlanner(injector=injector)
    exe = task.TaskExecutor()

    iTask = GraphIndexTask()
    plan = planner.plan(iTask)
    helper = task.TaskProgressHelper(plan)
    exe.executePlan(plan)

    with helper.output(iTask) as js:
        index = js.query()

    h_Set = [0, 1, 2, 5]

    # plan = nx.read_gpickle('./graph.pickle')

    graphs = []

    for k, v in tqdm(index['categories'].items()):

        graphs.extend(v)

    for g in graphs:
        iTask = ExtractKernelBagTask(g, 1, 5)
        plan = planner.plan(iTask, graph=plan)

    # exe.executePlan(plan)
    nx.write_gpickle(plan, './graph.pickle')
