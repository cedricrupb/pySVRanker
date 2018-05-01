from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from .bag_tasks import BagLoadingTask
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService
from .classification import common_tools, rank_y
from sklearn.grid_search import ParameterGrid
from .classification import BagNormalizeGramTask
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import math
import time


def pos(i, L):
    for j, l in enumerate(L):
        if i == l:
            return j
    return -1


def rank_to_mapping(tools, rank):
    pos = {}
    for i, t in enumerate(rank):
        pos[t] = i

    mapping = {}
    for i, t1 in enumerate(tools):
        for j, t2 in enumerate(tools):
            if i < j:
                if pos[t1] < pos[t2]:
                    mapping[(t1, t2)] = t1
                else:
                    mapping[(t1, t2)] = t2
    return mapping


def extract_labels(labels, t1, t2):
    out = {}
    for k, V in labels.items():
        if (t1, t2) in V:
            out[k] = V[(t1, t2)]
        else:
            out[k] = V[(t2, t1)]
    return out


def map_by_index(mapping, index):
    out = [None]*len(mapping)
    for k, v in mapping.items():
        out[index[k]] = v
    return out


class CzechLabelTask(Task):
    out_dir = Parameter('./czech/')

    def __init__(self, h, D):
        self.h = h
        self.D = D

    def require(self):
        return BagLoadingTask(self.h, self.D)

    def __taskid__(self):
        return 'CzechLabelTask_%d_%d' % (self.h, self.D)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            bag = i.query()

        index = list(bag.keys())
        labels = [bag[x]['label'] for x in index]
        del bag

        tools = common_tools(labels)
        ranks = rank_y(labels, tools)

        out = {}

        for i, k in enumerate(index):
            out[k] = rank_to_mapping(ranks[i])

        with self.output() as o:
            o.emit(out)


class CzechPairEvalTask(Task):
    out_dir = Parameter('./czech/')

    def __init__(self, first, second, C_Set, h, D,
                 train_index, test_index,
                 category=None, kernel='linear'):
        self.first = first
        self.second = second
        self.h = h
        self.D = D
        self.C_Set = C_Set
        self.category = category
        self.kernel = kernel
        self.train_index = train_index
        self.test_index = test_index

    def require(self):
        return [CzechLabelTask(self.h, self.D),
                BagNormalizeGramTask(self.h, self.D)]

    def __taskid__(self):
        return 'CzechPairEvalTask_%s' % (str(
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
            labels = i.query()

        labels = extract_labels(labels, self.first, self.second)

        with self.input()[i] as inp:
            D = inp.query()
            graphIndex = D['graphIndex']
            X = np.array(D['data'])
            del D

        y = map_by_index(labels, graphIndex)

        train_index = self.train_index
        test_index = self.test_index

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]

        params = {'kernel': 'precomputed', 'C': self.C_Set}
        svc = svm.SVC()
        clf = GridSearchCV(
            estimator=svc,
            param_grid=params,
            n_jobs=4,
            cv=10
        )
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        out = {
            'first': self.first,
            'second': self.second,
            'h': self.h,
            'D': self.D,
            'C': clf.best_params_['C'],
            'score': score
        }

        with self.output() as o:
            o.emit(out)


class CzechPairGridTask(Task):
    out_dir = Parameter('./czech/')

    def __init__(self, first, second, C_Set, h_Set, D_Set,
                 train_index, test_index,
                 category=None, kernel='linear'):
        self.first = first
        self.second = second
        self.h_Set = h_Set
        self.D_Set = D_Set
        self.C_Set = C_Set
        self.category = category
        self.kernel = kernel
        self.train_index = train_index
        self.test_index = test_index

    def require(self):
        paramGrid = {
            'h': self.h_Set,
            'D': self.D_Set
        }
        out = []
        for params in ParameterGrid(self.paramGrid):
            h = params['h']
            D = params['D']
            del params['h']
            del params['D']
            out.append(
                CzechPairEvalTask(
                    self.first,
                    self.second,
                    self.C_Set,
                    h,
                    D,
                    self.train_index,
                    self.test_index,
                    self.category,
                    self.kernel
                )
            )
        return out

    def __taskid__(self):
        return 'CzechPairGridTask_%s' % (str(
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
        best_setting = {'score': -math.inf}

        for inp in self.input():
            with inp as i:
                setting = i.query()

            if best_setting['score'] < setting['score']:
                best_setting = setting

        with self.output() as o:
            o.emit(best_setting)


class CzechPairPredictTask(Task):
    out_dir = Parameter('./czech/')

    def __init__(self, first, second, C_Set, h_Set, D_Set,
                 train_index, validate_index, test_index,
                 category=None, kernel='linear'):
        self.first = first
        self.second = second
        self.h_Set = h_Set
        self.D_Set = D_Set
        self.C_Set = C_Set
        self.category = category
        self.kernel = kernel
        self.train_index = train_index
        self.validate_index = validate_index
        self.test_index = test_index

    def require(self):
        out = [CzechPairGridTask(
            self.first, self.second, self.C_Set, self.h_Set, self.D_Set,
            self.train_index, self.validate_index, self.category, self.kernel
            ),
            CzechLabelTask(max(self.h_Set), max(self.D_Set))
        ]

        for h in self.h_Set:
            for D in self.D_Set:
                out.append(
                    BagNormalizeGramTask(
                        h, D, self.category, self.kernel
                    )
                )

        return out

    def __taskid__(self):
        return 'CzechPairPredictTask_%s' % (str(
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
            setting = i.query()

        with self.input()[0] as i:
            labels = i.query()

        labels = extract_labels(labels, self.first, self.second)

        hI = pos(setting['h'], self.h_Set)
        DI = pos(setting['D'], self.D_Set)
        P = hI * len(self.D_Set) + DI + 2

        with self.input()[P] as inp:
            D = inp.query()
            graphIndex = D['graphIndex']
            X = np.array(D['data'])
            del D

        y = map_by_index(labels, graphIndex)

        train_index = self.train_index
        test_index = self.test_index

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = svm.SVC(kernel='precomputed', probability=True, C=setting['C'])
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        prediction = clf.predict_proba(X_test)
        test_time = (time.time() - start_time) / len(y_test)

        prediction = [
            {k: v for (k, v) in zip(clf.classes_, x)}
            for x in prediction
        ]

        with self.output() as o:
            o.emit(
                {
                    'first': self.frist,
                    'second': self.second,
                    'train_time': train_time,
                    'test_time': test_time,
                    'predictions': prediction
                }
            )


class CzechPredictTask(Task):
    out_dir = Parameter('./czech/')

    def __init__(self, tools, C_Set, h_Set, D_Set,
                 train_index, validate_index, test_index,
                 category=None, kernel='linear'):
        self.tools = tools
        self.h_Set = h_Set
        self.D_Set = D_Set
        self.C_Set = C_Set
        self.category = category
        self.kernel = kernel
        self.train_index = train_index
        self.validate_index = validate_index
        self.test_index = test_index

    def require(self):
        out = []

        for i, t1 in enumerate(self.tools):
            for j, t2 in enumerate(self.tools):
                if i < j:
                    out.append(
                        CzechPairPredictTask(
                            t1,
                            t2,
                            self.C_Set,
                            self.h_Set,
                            self.D_Set,
                            self.train_index,
                            self.validate_index,
                            self.test_index,
                            self.category,
                            self.kernel
                        )
                    )

        return out

    def __taskid__(self):
        return 'CzechPredictTask_%s' % (str(
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

    @staticmethod
    def make_matrix(tool_prediction):
        index = np.array(list(tool_prediction.keys()))
        matrix = np.matrix([tool_prediction[k] for k in index])
        return index, matrix

    def run(self):
        tool_prediction = {}
        train_time = 0
        test_time = 0

        for inp in self.input():
            with inp as i:
                pair_eval = i.query()

            prediction = pair_eval['predictions']
            train_time += pair_eval['train_time']
            test_time += pair_eval['test_time']

            for k in [pair_eval['first'], pair_eval['second']]:

                pred = np.array([M[k] for M in prediction])

                if k not in tool_prediction:
                    tool_prediction[k] = pred
                else:
                    tool_prediction[k] += pred

        start_time = time.time()
        index, matrix = CzechPredictTask.make_matrix(tool_prediction)
        sort = matrix.argsort(axis=0)

        pred = [
            index[sort[i, :]].tolist() for i in range(sort.shape[0])
        ]
        test_time += (time.time() - start_time) / len(pred)

        with self.output() as o:
            o.emit(
                {
                    'tools': self.tools,
                    'train_time': train_time,
                    'test_time': test_time,
                    'predictions': pred
                }
            )
