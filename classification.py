from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


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

        for coord, c in self._times.items():
            index = []
            act_y = []
            for i, _y in enumerate(y):
                if coord in _y:
                    index.append(i)
                    act_y.append(_y[coord])
            c.fit(X[index:], act_y)

    def _incr(self, d, k):
        if k not in d:
            d[k] = 0
        d[k] += 1

    def predict_rank(self, X):
        prediction = {}

        for k, c in self._classifier.items():
            prediction[k] = c.predict(X)

        votes = []
        for _ in range(len(X)):
            votes.append({})

        for (i, j), c in self._times.items():
            t1 = self._tools[i]
            t2 = self._tools[j]
            p1 = prediction[t1]
            p2 = prediction[t2]
            tp = c.predict(X)

            for i, d in enumerate(votes):
                l1 = p1[i] == 'true'
                l2 = p2[i] == 'true'
                if l1 and not l2:
                    self._incr(d, t1)
                elif l2 and not l1:
                    self._incr(d, t2)
                else:
                    self._incr(d, tp[i])

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
