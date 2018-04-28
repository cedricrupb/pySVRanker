from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.model_selection import KFold
import random
from sklearn.base import clone
from sklearn.grid_search import ParameterGrid
import math
import time


def select_classifier(type_id):
    if type_id == 'majority':
        return ProgramRankMajority
    elif type_id == 'time':
        return ProgramRankPredictor
    elif type_id == 'quality':
        return QualityProgramRankPredictor
    else:
        raise ValueError('Unknown classifier type %s' % type_id)


def get_classifier_param_grid(type_id):
    if type_id == 'majority':
        return {}
    elif type_id == 'time':
        return {
            'C_solve': [0.0001, 0.01, 0.1, 1, 10, 100],
            'C_time': [0.0001, 0.01, 0.1, 1, 10, 100]
        }
    elif type_id == 'quality':
        return {
            'C_solve': [0.0001, 0.01, 0.1, 1, 10, 100]
        }
    else:
        raise ValueError('Unknown classifier type %s' % type_id)


def self_product(X):
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            if i < j:
                yield (x, y)


def incr(d, k, i=1):
    if k not in d:
        d[k] = 0
    d[k] += i


def enum_rank(y, tools=None):
    i = 0
    for t in y:
        if tools is not None and t not in tools:
            continue
        yield i, t
        i += 1


def rank_y(y, tools=None):
    out_y = []
    for _y in y:
        count = {}
        for i, t in enum_rank(_y, tools):
            for j, o in enum_rank(_y, tools):
                if i < j:
                    time_t = _y[t]['time']
                    time_o = _y[o]['time']

                    tbetter = time_t < time_o

                    solve_t = _y[t]['solve'] == 'correct'
                    solve_o = _y[o]['solve'] == 'correct'
                    if solve_t and not solve_o:
                        incr(count, t)
                        incr(count, o, 0)
                    elif solve_o and not solve_t:
                        incr(count, o)
                        incr(count, t, 0)
                    elif tbetter:
                        incr(count, t)
                        incr(count, o, 0)
                    else:
                        incr(count, o)
                        incr(count, t, 0)

        d = [x[0] for x in
             sorted(list(count.items()),
                    key=lambda y: y[1],
                    reverse=True)
             ]
        out_y.append(d)

    return out_y


def common_tools(y):
    tools = {}
    for _y in y:
        for t in _y.keys():
            if t not in tools:
                tools[t] = 0
            tools[t] += 1

    m = len(y)
    return [k for k, v in tools.items() if v == m]


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


class ProgramRankPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, C_solve=1.0, C_time=1.0):
        self.C_solve = C_solve
        self.C_time = C_time

    def _check_y(self, y):
        for _y in y:
            if len(_y) == 0:
                raise ValueError('Every point has to be labeled')
            for k, v in _y.items():
                if 'solve' not in v:
                    raise ValueError('Missing entry \'solve\'')
                if 'time' not in v:
                    raise ValueError('Missing entry \'time\'')

    def _init_classifier(self):
        self._classifier = {}
        self._time = {}

        for i, t in enumerate(self._tools):
            self._classifier[t] = MajorityOrSVC(self.C_solve)

            for j, o in enumerate(self._tools):
                if i < j:
                    self._time[(i, j)] = MajorityOrSVC(self.C_time)

    def _enum_y(self, y):
        i = 0
        for t in y:
            if t not in self._tools:
                continue
            yield (i, t)
            i += 1

    def _prep_y(self, y):
        out_y = []
        for _y in y:
            d = {}
            out_y.append(d)
            for i, t in self._enum_y(_y):
                d[t] = _y[t]['solve']
                for j, o in self._enum_y(_y):
                    if i < j:
                        t1 = _y[t]['time']
                        t2 = _y[o]['time']
                        if t1 > 900 and t2 > 900:
                            continue
                        if t1 < t2:
                            d[(i, j)] = t
                        else:
                            d[(i, j)] = o
        return out_y

    def fit(self, X, y):
        self._check_y(y)

        self._tools = common_tools(y)

        self._init_classifier()

        y = self._prep_y(y)

        for t, c in self._classifier.items():
            act_y = np.array([_y[t] for _y in y])
            c.fit(X, act_y)

        self._time_index = {}

        for coord, c in self._time.items():
            index = []
            act_y = []
            for i, _y in enumerate(y):
                if coord in _y:
                    index.append(i)
                    act_y.append(_y[coord])
            index = np.array(index, dtype=np.int)
            c.fit(X[index][:, index], act_y)
            self._time_index[coord] = index

    def _incr(self, d, k, i=1):
        if k not in d:
            d[k] = 0
        d[k] += i

    def predict_rank(self, X):
        prediction = {}

        for k, c in self._classifier.items():
            prediction[k] = c.predict(X)

        votes = []
        for _ in range(len(X)):
            votes.append({})

        for (i, j), c in self._time.items():
            t1 = self._tools[i]
            t2 = self._tools[j]
            p1 = prediction[t1]
            p2 = prediction[t2]
            tp = c.predict(X[:, self._time_index[(i, j)]])

            for i, d in enumerate(votes):
                l1 = p1[i] == 'correct'
                l2 = p2[i] == 'correct'
                if l1 and not l2:
                    self._incr(d, t1)
                    self._incr(d, t2, 0)
                elif l2 and not l1:
                    self._incr(d, t2)
                    self._incr(d, t1, 0)
                elif tp[i] == t1:
                    self._incr(d, t1)
                    self._incr(d, t2, 0)
                else:
                    self._incr(d, t2)
                    self._incr(d, t1, 0)

        y = [None] * len(X)

        for i, v in enumerate(votes):
            y[i] = [
                C[0] for C in sorted(
                    [(k, c) for k, c in v.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

        return y

    def predict(self, X):
        ranks = self.predict_rank(X)
        return [r[0] for r in ranks]

    def score(self, X, y):
        score = np.zeros(len(y))

        y_pred = self.predict(X)
        for i, _y in enumerate(y):
            if y_pred[i] == _y:
                score[i] = 1

        return score.mean()


class ProgramRankMajority(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def _check_y(self, y):
        for _y in y:
            for k, v in _y.items():
                if 'solve' not in v:
                    raise ValueError('Missing entry \'solve\'')
                if 'time' not in v:
                    raise ValueError('Missing entry \'time\'')

    def fit(self, X, y):
        self._check_y(y)

        tools = common_tools(y)

        y = rank_y(y, tools)
        print(tools)
        print(y)

        count = {}
        for _y in y:
            c = len(_y)-1
            for i, tool in enumerate(_y):
                self._incr(count, tool, c-i)

        self._rank = [x[0] for x in
                      sorted(list(count.items()),
                             key=lambda y: y[1],
                             reverse=True)
                      ]

    def _incr(self, d, k, i=1):
        if k not in d:
            d[k] = 0
        d[k] += i

    def predict_rank(self, X):
        return [self._rank] * len(X)

    def predict(self, X):
        ranks = self.predict_rank(X)
        return [r[0] for r in ranks]

    def score(self, X, y):
        score = np.zeros(len(y))

        y_pred = self.predict(X)
        for i, _y in enumerate(y):
            if y_pred[i] == _y:
                score[i] = 1

        return score.mean()


class QualityProgramRankPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, C_solve=1.0):
        self.C_solve = C_solve

    def _check_y(self, y):
        for _y in y:
            for k, v in _y.items():
                if 'solve' not in v:
                    raise ValueError('Missing entry \'solve\'')
                if 'time' not in v:
                    raise ValueError('Missing entry \'time\'')

    def _init_classifier(self):
        self._classifier = {}

        for i, t in enumerate(self._tools):
            self._classifier[t] = MajorityOrSVC(self.C_solve)

    def _prep_y(self, y):
        out_y = []
        for _y in y:
            d = {}
            out_y.append(d)
            for i, t in enumerate(_y):
                d[t] = _y[t]['solve']

        return out_y

    def fit(self, X, y):
        self._check_y(y)

        quality = {}

        self._tools = set([])
        for _y in y:
            for k, v in _y.items():
                self._tools.add(k)
                if v['solve'] == 'correct':
                    self._incr(quality, k)

        norm = len(y)
        self._quality = {k: (v/norm) for k, v in quality.items()}

        self._tools = list(self._tools)

        self._init_classifier()

        y = self._prep_y(y)

        for t, c in self._classifier.items():
            act_y = np.array([_y[t] for _y in y])
            c.fit(X, act_y)

    def _incr(self, d, k, i=1):
        if k not in d:
            d[k] = 0
        d[k] += i

    def predict_rank(self, X):
        prediction = {}

        for k, c in self._classifier.items():
            prediction[k] = c.predict(X)

        votes = []
        for _ in range(len(X)):
            votes.append({})

        for (t1, t2) in self_product(self._tools):
            p1 = prediction[t1]
            p2 = prediction[t2]

            for i, d in enumerate(votes):
                l1 = p1[i] == 'correct'
                l2 = p2[i] == 'correct'
                if l1 and not l2:
                    self._incr(d, t1)
                    self._incr(d, t2, 0)
                elif l2 and not l1:
                    self._incr(d, t2)
                    self._incr(d, t1, 0)
                else:
                    q_t1 = self._quality[t1]
                    q_t2 = self._quality[t2]
                    if q_t1 <= q_t2:
                        self._incr(d, t1)
                        self._incr(d, t2, 0)
                    else:
                        self._incr(d, t2)
                        self._incr(d, t1, 0)

        y = [None] * len(X)

        for i, v in enumerate(votes):
            y[i] = [
                C[0] for C in sorted(
                    [(k, c) for k, c in v.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
            ]

        return y

    def predict(self, X):
        ranks = self.predict_rank(X)
        return [r[0] for r in ranks]

    def score(self, X, y):
        score = np.zeros(len(y))

        y_pred = self.predict(X)
        for i, _y in enumerate(y):
            if y_pred[i] == _y:
                score[i] = 1

        return score.mean()
