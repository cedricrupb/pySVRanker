from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from .bags.model import read_bag
from sklearn.model_selection import KFold
import random
from sklearn.base import clone
from sklearn.grid_search import ParameterGrid
import math
import time


def incr(d, k, i=1):
    if k not in d:
        d[k] = 0
    d[k] += i


def rank_y(y):
    out_y = []
    for _y in y:
        count = {}
        for i, t in enumerate(_y):
            for j, o in enumerate(_y):
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

    def _prep_y(self, y):
        out_y = []
        for _y in y:
            d = {}
            out_y.append(d)
            for i, t in enumerate(_y):
                d[t] = _y[t]['solve']
                for j, o in enumerate(_y):
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

        self._tools = set([])
        for _y in y:
            for k in _y:
                self._tools.add(k)

        self._tools = list(self._tools)

        self._init_classifier()

        y = self._prep_y(y)

        for t, c in self._classifier.items():
            act_y = np.array([_y[t] for _y in y])
            c.fit(X, act_y)

        for coord, c in self._time.items():
            index = []
            act_y = []
            for i, _y in enumerate(y):
                if coord in _y:
                    index.append(i)
                    act_y.append(_y[coord])
            index = np.array(index, dtype=np.int)
            c.fit(X[index][:, index], act_y)
            self._time_index = index

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
            tp = c.predict(X[:, self._time_index])

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

        y = rank_y(y)

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


def kfold(clf, X, y, folds=10, shuffle=True):
    scores = []
    loo = KFold(10, shuffle=shuffle, random_state=random.randint(0, 100))
    for train_index, test_index in loo.split(np.arange(len(y))):

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]

        Clf = clone(clf)
        clf.fit(X_train, y_train)

        y_test = [_y[0] for _y in rank_y(y_test)]
        score = clf.score(X_test, y_test)
        scores.append(score)

    return np.mean(scores), np.std(scores)


def grid_prediction_perform(X, y, C=[0.001, 0.01, 0.1, 1, 10, 100, 1000], folds=10, shuffle=True):
    param_grid = {'C_solve': C,
                  'C_time': C}

    max_mean = -math.inf
    buff_std = 0
    max_param = None
    for params in ParameterGrid(param_grid):

        clf = ProgramRankPredictor(C_solve=params['C_solve'],
                                   C_time=params['C_time'])

        mean, std = kfold(clf, X, y, folds, shuffle)

        if mean > max_mean:
            max_mean = mean
            buff_std = std
            max_param = params

    return max_param, max_mean, buff_std


if __name__ == '__main__':
    path = '/Users/cedricrichter/Documents/Arbeit/Ranking/bootstrap-scripts/gram/ExtractKernelEntitiesTask_0_5_-1159164113_time.json'
    path_1 = '/Users/cedricrichter/Documents/Arbeit/Ranking/bootstrap-scripts/gram/ExtractKernelEntitiesTask_1_5_-1159164113_time.json'

    bag_0 = read_bag(path)
    bag_1 = read_bag(path_1)

    bag = bag_0 + bag_1

    gram_time = time.time()
    gI, X = bag.normalized_gram()
    gram_time = (time.time() - gram_time) / len(gI)

    y = np.array(bag.indexed_labels(gI))
    times = bag.indexed_times(gI)

    scores = []
    test_times = []
    train_times = []
    loo = KFold(10, shuffle=True, random_state=random.randint(0, 100))
    for train_index, test_index in loo.split(np.arange(len(y))):

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]

        train_time = 0.0
        for index in train_index:
            train_time += times[index] + gram_time

        params, mean, std = grid_prediction_perform(X_train, y_train)
        print(params)
        print('%0.2f (Std: %0.4f)' % (mean, 2*std))

        clf = ProgramRankPredictor(C_solve=params['C_solve'],
                                   C_time=params['C_time'])
        start = time.time()
        clf.fit(X_train, y_train)
        train_time += (time.time() - start)

        start = time.time()
        y_test = [_y[0] for _y in rank_y(y_test)]
        score = clf.score(X_test, y_test)
        test_time_clf = (time.time() - start) / len(y)

        test_time = []
        for index in test_index:
            test_time.append(times[index] + gram_time + test_time_clf)

        scores.append(score)
        train_times.append(train_time)
        test_times.extend(test_time)

    print('Test: %0.2f (Std: %0.4f)' % (np.mean(scores), 2*np.std(scores)))
    print('Train time %f (Std: %2.2f)' % (np.mean(train_times), 2*np.std(train_times)))
    print('Test time %f (Std: %2.2f)' % (np.mean(test_times), 2*np.std(test_times)))
