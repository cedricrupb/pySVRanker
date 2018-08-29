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
import time
import math
from .rank_scores import select_score
from .bag_tasks import BagLabelMatrixTask, index, reverse_index, ranking
from sklearn.base import BaseEstimator, ClassifierMixin


class BorderMajorityClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        count = {}

        for _y in y:
            for i, r in enumerate(_y):
                value = len(_y) - (i + 1)
                if r not in count:
                    count[r] = 0
                count[r] += value

        ranking = sorted([(k, c) for k, c in count.items()], key=lambda x: x[1],
                         reverse=True)
        self.ranking = [k for k, c in ranking]

    def predict_rank(self, X):
        return [self.ranking] * len(X)

    def predict(self, X):
        return self.predict_rank(X)[0]


class MajorityEvaluationTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, scores, train_index, test_index,
                 category=None, task_type=None):
                self.scores = scores
                self.train_index = train_index
                self.test_index = test_index
                self.category = category
                self.task_type = task_type

    def require(self):
        out = [BagGraphIndexTask(0, 5,
                                 self.category, self.task_type),
               BagFilterTask(0, 5,
                             self.category, self.task_type),
               BagLabelMatrixTask(0, 5,
                                  self.category, self.task_type)]
        return out

    def __taskid__(self):
        return 'MajorityEvaluationTask_%s' % (str(
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

        rank_train = [y[i] for i in self.train_index]

        clf = BorderMajorityClassifier()
        clf.fit(None, rank_train)

        rank_pred = clf.predict_rank(self.test_index)

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
                    'result': empirical,
                    'raw_results': raw_empircal
                }
            )


class CVMajorityEvalutionTask(Task):
        out_dir = Parameter('./eval/')
        k = Parameter(10)

        def __init__(self, scores, full_index,
                     category=None, task_type=None):
            self.scores = scores
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
                MajorityEvaluationTask(
                    self.scores,
                    train_index.tolist(),
                    test_index.tolist(),
                    self.category,
                    self.task_type
                )
                for train_index, test_index in loo.split(index)
            ]

        def __taskid__(self):
            return 'CVMajorityEvalutionTask_%s' % (str(
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

            with self.output() as o:
                o.emit(
                    {
                        'param': self.get_params(),
                        'results': results
                    }
                )
