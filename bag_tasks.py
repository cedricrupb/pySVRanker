import matplotlib
matplotlib.use('Agg')
from .bag import ProgramBags, read_bag, normalize_gram, enumerateable, indexMap
from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService, FileTarget
from .gram_tasks import ExtractKernelEntitiesTask
from .kernel_function import select_full
import numpy as np
from .classification import select_classifier, rank_y
from .rank_scores import select_score
import math
import time
from sklearn.model_selection import KFold
import random
from sklearn.grid_search import ParameterGrid
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import os
from scipy.sparse import issparse


def is_dict(D):
    try:
        D.items()
        return True
    except AttributeError:
        return False


def mean_std(L):
    O = {}
    for k in L[0]:
        coll = [l[k] for l in L]
        if is_dict(coll[0]):
            coll = mean_std(coll)
        else:
            coll = {
                'mean': np.mean(coll),
                'median': np.median(coll),
                'std': np.std(coll)
            }
        O[k] = coll

    return O


def dominates(A, B):
    for i, a in enumerate(A):
        if a < B[i]:
            return False

    return True


def pareto_front(knownSet, entity, key):
    front = []
    entity_add = True
    entity_k = key(entity)
    for knownEntity in knownSet:
        k = key(knownEntity)
        if dominates(k, entity_k):
            return knownSet.copy()
        elif not dominates(entity_k, k):
            front.append(knownEntity)

    if entity_add:
        front.append(entity)

    return front


class BagLoadingTask(Task):
    out_dir = Parameter('./gram/')
    pattern = Parameter('./task_%d_%d.json')

    def __init__(self, h, D):
        self.h = h
        self.D = D

    def require(self):
        return None

    def __taskid__(self):
        return 'BagLoadingTask_%d_%d' % (self.h, self.D)

    def output(self):
        src_path = self.pattern.value % (self.h, self.D)
        return CachedTarget(
            LocalTarget(src_path, service=JsonService)
        )

    def run(self):
        pass


class BagGraphIndexTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D):
        self.h = h
        self.D = D

    def require(self):
        return BagLoadingTask(self.h, self.D)

    def __taskid__(self):
        return 'BagGraphIndexTask_%d' % (self.D)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            D = i.query()

        out = {}

        for k in D:
            indexMap(k, out)

        with self.output() as o:
            o.emit(out)


class BagFeatureTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None):
        self.h = h
        self.D = D
        self.category = category

    def require(self):
        return [BagGraphIndexTask(self.h,
                                  self.D),
                BagLoadingTask(self.h, self.D)]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = '_'.join(enumerateable(self.category))

        return 'BagFeatureTask_%d_%d_%s' % (self.h, self.D, cat)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        with self.input()[1] as i:
            bag = ProgramBags(content=i.query())

        bag.graphIndex = graphIndex

        if self.category is not None:
            bag = bag.get_category(self.category)

        features = bag.features()

        NZ = features.nonzero()
        data = features[NZ].A
        shape = features.get_shape()

        out = {
            'graphIndex': bag.graphIndex,
            'nodeIndex': bag.nodeIndex,
            'rows': NZ[0].tolist(),
            'columns': NZ[1].tolist(),
            'data': data.tolist(),
            'row_shape': shape[0],
            'column_shape': shape[1]
        }

        with self.output() as o:
            o.emit(out)


class BagGramTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.kernel = kernel

    def require(self):
        return [BagGraphIndexTask(self.h,
                                  self.D),
                BagLoadingTask(self.h, self.D)]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = '_'.join(enumerateable(self.category))

        return 'BagGramTask_%d_%d_%s_%s' % (self.h, self.D, self.kernel, cat)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        with self.input()[1] as i:
            bag = ProgramBags(content=i.query())

        bag.graphIndex = graphIndex

        if self.category is not None:
            bag = bag.get_category(self.category)

        if self.kernel == 'linear':
            gram = bag.gram().toarray()
        else:
            kernel = select_full(self.kernel)
            if kernel is None:
                raise ValueError('Unknown kernel %s' % self.kernel)
            gram = bag.gram(kernel=kernel)
            if issparse(gram):
                gram = gram.toarray()

        data = gram.tolist()

        out = {
            'graphIndex': bag.graphIndex,
            'data': data
        }

        with self.output() as o:
            o.emit(out)


class BagSumGramTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, hSet, D, category=None, kernel='linear'):
        self.hSet = hSet
        self.D = D
        self.category = category
        self.kernel = kernel

    def require(self):
        return [
            BagGramTask(h, self.D, self.category, self.kernel)
            for h in self.hSet
        ]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = '_'.join(enumerateable(self.category))

        return 'BagSumGramTask_%s_%d_%s_%s' % (str(
                                                      containerHash(self.hSet)
                                                      ),
                                               self.D, self.kernel, cat
                                               )

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        GR = None

        for inp in self.input():
            print(inp.path)
            with inp as i:
                D = i.query()
                gI = D['graphIndex']
                gram = np.array(D['data'])
                del D
            if GR is None:
                GR = gram
            else:
                GR += gram
            del gram

        data = GR.tolist()

        out = {
            'graphIndex': gI,
            'data': data
        }

        with self.output() as o:
            o.emit(out)


class BagNormalizeGramTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.kernel = kernel

    def require(self):
        hSet = [h for h in enumerateable(self.h)]
        if len(hSet) == 1:
            return BagGramTask(hSet[0], self.D,
                               self.category, self.kernel)
        else:
            return BagSumGramTask(hSet, self.D,
                                  self.category, self.kernel)

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = '_'.join(enumerateable(self.category))

        return 'BagNormGramTask_%s_%d_%s_%s' % (str(
                                                      containerHash(self.h)
                                                      ),
                                                self.D, self.kernel, cat
                                                )

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            D = i.query()
            graphIndex = D['graphIndex']
            gram = np.array(D['data'])
            del D

        data = normalize_gram(gram).tolist()

        out = {
            'graphIndex': graphIndex,
            'data': data
        }

        with self.output() as o:
            o.emit(out)


class BagClassifierEvalutionTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, clf_type, clf_params,
                 h, D, scores,
                 train_index, test_index,
                 category=None,
                 kernel='linear'):
        self.clf_type = clf_type
        self.clf_params = clf_params
        self.kernel = kernel
        self.h = h
        self.D = D
        self.scores = scores
        self.train_index = train_index
        self.test_index = test_index
        self.category = category

    def require(self):
        h = [h for h in range(self.h+1)]
        return [BagLoadingTask(self.h, self.D),
                BagNormalizeGramTask(h, self.D, self.category,
                                     self.kernel)]

    def __taskid__(self):
        return 'BagClassifierEvalutionTask_%s' % (str(
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

    def _build_classifier(self):
        clf = select_classifier(self.clf_type)
        return clf(**self.clf_params)

    def _build_maps(self):
        with self.input()[0] as i:
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
        V = [
            m for m in sorted(list(mapping.items()), key=lambda x: index[x[0]])
        ]
        graphs = [m[0] for m in V]
        return graphs, np.array([m[1] for m in V])

    def run(self):
        with self.input()[1] as i:
            D = i.query()
            graphIndex = D['graphIndex']
            X = np.array(D['data'])
            del D

        y, times = self._build_maps()
        scores = self._build_score(y, times)
        graphs, y = BagClassifierEvalutionTask._index_map(graphIndex, y)

        train_index = self.train_index
        test_index = self.test_index

        X_train, X_test = X[train_index][:, train_index], X[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]
        y_test = rank_y(y_test)

        clf = self._build_classifier()
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        prediction = clf.predict_rank(X_test)
        test_time = (time.time() - start_time) / len(y_test)

        empirical = {}
        for i, pred in enumerate(prediction):
            expected = y_test[i]
            g = graphs[test_index[i]]
            for k, score in scores.items():
                if k not in empirical:
                    empirical[k] = 0.0
                empirical[k] += score(pred, expected, g) / len(y_test)

        with self.output() as emitter:
            emitter.emit(
                {
                    'parameter': self.get_params(),
                    'train_time': train_time,
                    'test_time': test_time,
                    'result': empirical
                }
            )


class BagKFoldTask(Task):
    k = Parameter(10)
    out_dir = Parameter('./eval/')

    def __init__(self, clf_type, clf_params,
                 graph_count, h, D, scores,
                 subset_index=None,
                 category=None,
                 kernel='linear'):
        self.clf_type = clf_type
        self.clf_params = clf_params
        self.kernel = kernel
        self.graph_count = graph_count
        self.h = h
        self.D = D
        self.scores = scores
        self.subset_index = subset_index
        self.category = category

    def _index(self):
        if self.subset_index is None:
            return [x for x in range(self.graph_count)]
        else:
            return self.subset_index

    def require(self):
        index = np.array(self._index())
        loo = KFold(self.k.value, shuffle=True, random_state=random.randint(0, 100))
        return [
            Optional(BagClassifierEvalutionTask(
                self.clf_type,
                self.clf_params,
                self.h,
                self.D,
                self.scores,
                train_index.tolist(),
                test_index.tolist(),
                self.category,
                self.kernel
            ))
            for train_index, test_index in loo.split(index)
        ]

    def __taskid__(self):
        return 'BagKFoldTask_%s' % (str(
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
        results = []
        for inp in self.input():
            if inp is None:
                continue
            with inp as i:
                R = i.query()
                if 'parameter' in R:
                    del R['parameter']
                results.append(R)

        R = mean_std(results)

        with self.output() as emitter:
            emitter.emit(
                {
                    'parameter': self.get_params(),
                    'train_time': R['train_time'],
                    'test_time': R['test_time'],
                    'result': R['result']
                }
            )


class BagParameterGridTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, clf_type, paramGrid,
                 graph_count, scores, opt_scores,
                 subset_index=None,
                 category=None,
                 kernel='linear'):
        self.clf_type = clf_type
        self.paramGrid = paramGrid
        self.kernel = kernel
        self.graph_count = graph_count
        self.scores = scores
        self.opt_scores = opt_scores
        self.subset_index = subset_index
        self.category = category

    def require(self):
        out = []
        for params in ParameterGrid(self.paramGrid):
            h = params['h']
            D = params['D']
            del params['h']
            del params['D']
            out.append(
                BagKFoldTask(
                    self.clf_type,
                    params,
                    self.graph_count,
                    h,
                    D,
                    self.scores,
                    self.subset_index,
                    self.category,
                    self.kernel
                )
            )
        return out

    def __taskid__(self):
        return 'BagParameterGridTask_%s' % (str(
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
        pareto = []

        def score_key(result):
            o = []
            for k in self.opt_scores:
                o.append(result['result'][k]['mean'])
            return o

        for inp in self.input():
            if inp is None:
                continue
            with inp as i:
                res = i.query()
            pareto = pareto_front(pareto, res, score_key)

        with self.output() as o:
            o.emit(pareto)


class BagKParamGridTask(Task):
    k = Parameter(10)
    out_dir = Parameter('./eval/')

    def __init__(self, clf_type, paramGrid,
                 graph_count, scores, opt_scores,
                 subset_index=None,
                 category=None,
                 kernel='linear'):
        self.clf_type = clf_type
        self.paramGrid = paramGrid
        self.kernel = kernel
        self.graph_count = graph_count
        self.scores = scores
        self.opt_scores = opt_scores
        self.subset_index = subset_index
        self.category = category

    def _index(self):
        if self.subset_index is None:
            return [x for x in range(self.graph_count)]
        else:
            return self.subset_index

    def require(self):
        index = np.array(self._index())
        loo = KFold(self.k.value, shuffle=True, random_state=random.randint(0, 100))
        return [
            BagParameterGridTask(
                self.clf_type,
                self.paramGrid,
                self.graph_count,
                self.scores,
                self.opt_scores,
                subset_index=train_index.tolist(),
                category=self.category,
                kernel=self.kernel
            )
            for train_index, test_index in loo.split(index)
        ]

    def __taskid__(self):
        return 'BagKGridTask_%s' % (str(
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
        results = []
        for inp in self.input():
            if inp is None:
                continue
            with inp as i:
                results.append(i.query())

        with self.output() as emitter:
            emitter.emit(results)


class BagMDSTask(Task):
        out_dir = Parameter('./gram/')

        def __init__(self, h, D, category=None, kernel='linear'):
            self.h = h
            self.D = D
            self.category = category
            self.kernel = kernel

        def require(self):
            h = [h for h in range(self.h+1)]
            return [BagLoadingTask(self.h, self.D),
                    BagNormalizeGramTask(h, self.D, self.category,
                                         self.kernel)]

        def __taskid__(self):
            cat = 'all'
            if self.category is not None:
                cat = '_'.join(enumerateable(self.category))

            return 'BagMDSTask_%d_%d_%s_%s' % (self.h, self.D, self.kernel,
                                               cat
                                               )

        def output(self):
            return FileTarget(self.out_dir.value+self.__taskid__()+'.png')

        def _build_maps(self):
            with self.input()[0] as i:
                D = i.query()
            map_to_labels = {k: v['label'] for k, v in D.items()}
            map_to_times = {k: v['time'] if 'time' in v else math.inf for k, v in D.items()}
            del D
            return map_to_labels, map_to_times

        @staticmethod
        def _index_map(index, mapping):
            V = [
                m[1] for m in sorted(list(mapping.items()), key=lambda x: index[x[0]])
            ]
            return np.array(V)

        @staticmethod
        def _state_time_split(y):
            y_solve = np.array(['UNKNOWN']*len(y))
            y_time = np.array(['UNKNOWN']*len(y))

            for i, d in enumerate(y):
                solve = [k for k in d if d[k]['solve'] == 'correct']
                if len(solve) == 1:
                    y_solve[i] = solve[0]
                time_rank = sorted(list(d.keys()), key=lambda x: d[x]['time'])
                y_time[i] = time_rank[0]

            return y_solve, y_time

        def run(self):
            with self.input()[1] as i:
                D = i.query()
                graphIndex = D['graphIndex']
                X = np.array(D['data'])
                del D

            y, times = self._build_maps()
            y = BagMDSTask._index_map(graphIndex, y)
            y_solve, y_time = BagMDSTask._state_time_split(y)

            dis = np.ones(X.shape, dtype=X.dtype) - X
            colors = ['grey', 'green', 'red']
            tName = ['UNKNOWN', 'IUV', 'ESBMC']
            aName = ['UNKNOWN', 'Tester', 'Verificator']

            l_solve = np.zeros(len(y_solve))
            l_time = np.zeros(len(y_time))

            for i in range(len(l_solve)):
                for j, t in enumerate(tName):
                    if y_solve[i] == t:
                        l_solve[i] = j
                    if y_time[i] == t:
                        l_time[i] = j

            mds = MDS(n_components=2, dissimilarity="precomputed", n_init=10)
            X_r = mds.fit_transform(dis)
            stress = mds.stress_

            plt.figure(1)
            plt.suptitle('MDS of GRAM dataset (h: %s, D: %s) [%s points] (Stress: %2.2f)' %
                         (str(self.h), str(self.D), str(len(X_r)), stress))

            plt.subplot(121)
            for color, i, t in zip(colors, range(len(aName)), aName):
                plt.scatter(X_r[l_solve == i, 0], X_r[l_solve == i, 1],
                            color=color, alpha=.8,
                            lw=2,
                            label=t)
            plt.legend(loc='best', shadow=False, scatterpoints=1)

            plt.subplot(122)
            for color, i, t in zip(colors, range(len(aName)), aName):
                plt.scatter(X_r[l_time == i, 0], X_r[l_time == i, 1],
                            color=color, alpha=.8,
                            lw=2,
                            label=t)
            plt.legend(loc='best', shadow=False, scatterpoints=1)

            path = self.output().path

            directory = os.path.dirname(path)

            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(path)
            plt.close()
