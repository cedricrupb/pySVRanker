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
from scipy.sparse import issparse, coo_matrix
from .prepare_tasks import select_svcomp
import re
from .svcomp15 import MissingPropertyTypeException
from sklearn.decomposition import KernelPCA
import json
from scipy.sparse import issparse


def is_correct(label):
    return label['solve'] == 1


def is_false(label):
    return label['solve'] == -1


def is_faster(labelA, labelB, major=False):
    if labelB['time'] >= 900:
        return labelA['time'] < 900 or major
    return labelA['time'] < labelB['time']


def index(x, y, n):
    if x >= n:
        raise ValueError('x: %d is out of range 0 to %d' % (x, n))
    if y >= n:
        raise ValueError('y: %d is out of range 0 to %d' % (y, n))
    if y == x:
        return x
    if x > y:
        tmp = y
        y = x
        x = tmp

    return int(x * (n - 0.5*(x+1))
               + (y - (x+1))
               + n)


def reverse_index(i, n):
    if i < n:
        return i, i

    i = i - n + 1
    x = 0
    while int(x * (n - 0.5*(x+1))) < i:
        x += 1
    x -= 1
    y = i - int(x * (n - 0.5*(x+1))) + x
    return x, y


def is_dict(D):
    try:
        D.items()
        return True
    except AttributeError:
        return False


def mean_std(L):
    O = {}
    for k in L[0]:
        if 'raw' in k:
            continue
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


def is_better(ci, cj, fij):
    return ci > cj or (ci == cj and fij == 1)


def borda_major(bag):
    votes = {}
    for k, B in bag.items():
        for tool_x, label_x in B['label'].items():
            for tool_y, label_y in B['label'].items():
                if tool_x < tool_y:
                    if tool_x not in votes:
                        votes[tool_x] = 0
                    if tool_y not in votes:
                        votes[tool_y] = 0
                    ci =\
                        1 if is_correct(label_x) else \
                        (-1 if is_false(label_x) else 0)

                    cj =\
                        1 if is_correct(label_y) else \
                        (-1 if is_false(label_y) else 0)

                    fij = 1 if is_faster(label_x, label_y) else 0
                    fij = -1 if fij == 0 and is_faster(label_x, label_y) else fij
                    if max([ci, cj, fij]) == min([ci, cj, fij]) and fij == 0:
                        continue
                    votes[tool_x if is_better(ci, cj, fij) else tool_y] += 1

    votes = [t[0] for t in sorted(list(votes.items()), key=lambda X: X[1], reverse=True)]
    return {k: i for i, k in enumerate(votes)}


def ranking(row, n):
    N = np.zeros(n)

    for i in range(n):
        correct_i = row[index(i, i, n)]
        for j in range(n):
            if i < j:
                correct_j = row[index(j, j, n)]
                faster_i = row[index(i, j, n)]
                N[i if is_better(correct_i, correct_j, faster_i) else j] += 1

    return N.argsort()[::-1]


class FeatureJsonService:

    def __init__(self, s):
        self.__src = s

    def emit(self, obj):
        for k, V in obj.items():
            if issparse(V):
                NZ = V.nonzero()
                data = V[NZ].A
                shape = V.get_shape()

                obj[k] = {
                    'sparse': True,
                    'rows': NZ[0].tolist(),
                    'columns': NZ[1].tolist(),
                    'data': data.tolist()[0],
                    'row_shape': shape[0],
                    'column_shape': shape[1]
                }

        json.dump(obj, self.__src, indent=4)

    def query(self):
        obj = json.load(self.__src)

        for k, V in obj.items():
            if 'sparse' in V:
                obj[k] = coo_matrix((V['data'], (V['rows'], V['columns'])),
                                    shape=(V['row_shape'],
                                    V['column_shape'])).tocsr()

        return obj

    def isByte(self):
        return False


class BagLoadingTask(Task):
    pattern = Parameter('./task_%d_%d.json')

    def __init__(self, h, D):
        self.h = h
        self.D = D

    def require(self):
        return None

    def __taskid__(self):
        return 'BagLoadingTask_%d_%d' % (self.h, self.D)

    def output(self):
        try:
            src_path = self.pattern.value % (self.h, self.D)
        except TypeError:
            print('Source is not formattable: %s. Continue without parameter.'
                  % self.pattern.value)
            src_path = self.pattern.value
        return CachedTarget(
            LocalTarget(src_path, service=JsonService)
        )

    def run(self):
        pass


class BagFilterTask(Task):
    out_dir = Parameter('./gram/')
    svcomp = Parameter('svcomp18')

    def __init__(self, h, D, category=None, task_type=None, by_id=False):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.by_id = by_id

    def _init_filter(self):
        categories = set(enumerateable(self.category))
        self._svcomp = select_svcomp(self.svcomp.value)
        prop = self._svcomp.select_propety(self.task_type)

        def filter(category, property):
            print(property)
            if prop is not None and prop not in property:
                return False

            if self.category is None:
                return True

            return category in categories
        self._filter = filter

    def detect_property(self, path):
        try:
            return self._svcomp.set_of_properties(path)
        except MissingPropertyTypeException:
            print('Problem with property. Ignore')
            return None

    def detect_category(self, path):
        reg = re.compile('sv-benchmarks\/c\/[^\/]+\/')
        o = reg.search(path)
        if o is None:
            return 'unknown'
        return o.group()[16:-1]

    def require(self):
        return BagLoadingTask(self.h, self.D)

    def __taskid__(self):
        s = 'BagFilterTask_%d_%d' % (self.h, self.D)
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

    def run(self):
        self._init_filter()
        with self.input()[0] as i:
            B = i.query()

        D = set([])
        for name, V in B.items():
            if self.by_id:
                f = name
            else:
                f = V['file']
            if not self._filter(
                self.detect_category(f),
                self.detect_property(f)
            ):
                D.add(name)

        B = {b: V for b, V in B.items() if b not in D}

        with self.output() as o:
            o.emit(B)


class BagGraphIndexTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, task_type=None):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type

    def require(self):
        return BagFilterTask(self.h, self.D,
                             self.category, self.task_type)

    def __taskid__(self):
        s = 'BagGraphIndexTask_%d' % (self.D)
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

    def run(self):
        with self.input()[0] as i:
            D = i.query()

        out = {}

        for k in D:
            indexMap(k, out)

        with self.output() as o:
            o.emit(out)


class BagLabelMatrixTask(Task):
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
        s = 'BagLabelMatrixTask_%d_%d' % (self.h, self.D)
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
        tools = [k for k, v in C.items() if v == F]

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
        self.major = borda_major(bag)

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
            for tool_x, label_x in B['label'].items():
                index_x = tool_index[tool_x]
                label_matrix[index_g, index(index_x, index_x, n)] =\
                    1 if is_correct(label_x) else \
                    (-1 if is_false(label_x) else 0)

                for tool_y, label_y in B['label'].items():
                    index_y = tool_index[tool_y]
                    if index_y > index_x:
                        major = self.major[tool_x] < self.major[tool_y]
                        v = 1 if is_faster(label_x, label_y, major) else 0
                        v = -1 if v == 0 and is_faster(label_x, label_y, major) else v
                        label_matrix[index_g, index(index_x, index_y, n)] =\
                            v
            rank = ranking(label_matrix[index_g, :], n)
            rankings[index_g] = [self.tools[i] for i in rank]

        with self.output() as o:
            o.emit(
                {
                    'tools': self.tools,
                    'label_matrix': label_matrix.tolist(),
                    'rankings': rankings.tolist()
                }
            )


class BagFeatureTask(Task):
    out_dir = Parameter('./gram/')
    svcomp = Parameter('svcomp15')

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
        cat = 'all'
        if self.category is not None:
            cat = str(containerHash(self.category))

        tt = ''
        if self.task_type is not None:
            tt = '_'+str(self.task_type)

        return 'BagFeatureTask_%d_%d_%s' % (self.h, self.D, cat)\
               + tt

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=FeatureJsonService)
        )

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        with self.input()[1] as i:
            bag = ProgramBags(content=i.query(), svcomp=self.svcomp.value)

        bag.graphIndex = graphIndex

        features = bag.features()

        out = {
            'graphIndex': bag.graphIndex,
            'nodeIndex': bag.nodeIndex,
            'features': features
        }

        with self.output() as o:
            o.emit(out)


class BagGramTask(Task):
    out_dir = Parameter('./gram/')
    svcomp = Parameter('svcomp15')

    def __init__(self, h, D, category=None, task_type=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        return [BagGraphIndexTask(self.h,
                                  self.D,
                                  self.category, self.task_type),
                BagFilterTask(self.h, self.D,
                              self.category, self.task_type)]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = str(containerHash(self.category))

        tt = ''
        if self.task_type is not None:
            tt = '_'+str(self.task_type)

        return 'BagGramTask_%d_%d_%s_%s' % (self.h, self.D, self.kernel, cat)\
               + tt

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()

        with self.input()[1] as i:
            bag = ProgramBags(content=i.query(), svcomp=self.svcomp.value)

        bag.graphIndex = graphIndex
        print(bag.graphIndex['counter'])

        if self.kernel == 'linear':
            gram = bag.gram().toarray()
        else:
            kernel = select_full(self.kernel)
            if kernel is None:
                raise ValueError('Unknown kernel %s' % self.kernel)
            gram = bag.gram(kernel=kernel)
            if issparse(gram):
                gram = gram.toarray()

        print(gram.shape)
        data = gram.tolist()

        out = {
            'graphIndex': bag.graphIndex,
            'data': data
        }

        with self.output() as o:
            o.emit(out)


class BagSumGramTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, hSet, D, category=None,
                 task_type=None, kernel='linear'):
        self.hSet = hSet
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        return [
            BagGramTask(h, self.D, self.category, self.task_type, self.kernel)
            for h in self.hSet
        ]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = str(containerHash(self.category))

        tt = ''
        if self.task_type is not None:
            tt = '_'+str(self.task_type)

        return 'BagSumGramTask_%s_%d_%s_%s' % (str(
                                                      containerHash(self.hSet)
                                                      ),
                                               self.D, self.kernel, cat
                                               )\
            + tt

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        GR = None

        for inp in self.input():
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

    def __init__(self, h, D, category=None, task_type=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        hSet = [h for h in enumerateable(self.h)]
        if len(hSet) == 1:
            return BagGramTask(hSet[0], self.D,
                               self.category, self.task_type, self.kernel)
        else:
            return BagSumGramTask(hSet, self.D,
                                  self.category, self.task_type, self.kernel)

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = str(containerHash(self.category))

        tt = ''
        if self.task_type is not None:
            tt = '_'+str(self.task_type)

        return 'BagNormGramTask_%s_%d_%s_%s' % (str(
                                                      containerHash(self.h)
                                                      ),
                                                self.D, self.kernel, cat
                                                )\
            + tt

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
                 task_type=None,
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
        self.task_type = task_type

    def require(self):
        h = [h for h in range(self.h+1)]
        return [BagFilterTask(self.h, self.D,
                              self.category, self.task_type),
                BagNormalizeGramTask(h, self.D, self.category, self.task_type,
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
        mapping = {k: v for k, v in mapping.items() if k in index}
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
        times['train'] = train_time

        start_time = time.time()
        prediction = clf.predict_rank(X_test)
        test_time = (time.time() - start_time) / len(y_test)
        times['prediction'] = test_time

        empirical = {}
        raw_empircal = {}
        for i, pred in enumerate(prediction):
            expected = y_test[i]
            g = graphs[test_index[i]]
            for k, score in scores.items():
                if k not in empirical:
                    empirical[k] = 0.0
                    raw_empircal[k] = []
                s = score(pred, expected, g)
                empirical[k] += s / len(y_test)
                raw_empircal[k].append(s)

        with self.output() as emitter:
            emitter.emit(
                {
                    'parameter': self.get_params(),
                    'train_time': train_time,
                    'test_time': test_time,
                    'result': empirical,
                    'raw_results': raw_empircal
                }
            )


class BagKFoldTask(Task):
    k = Parameter(10)
    out_dir = Parameter('./eval/')
    random_state = Parameter(0)

    def __init__(self, clf_type, clf_params,
                 graph_count, h, D, scores,
                 subset_index=None,
                 category=None,
                 task_type=None,
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
        self.task_type = task_type

    def _index(self):
        if self.subset_index is None:
            return [x for x in range(self.graph_count)]
        else:
            return self.subset_index

    def require(self):
        index = np.array(self._index())
        loo = KFold(self.k.value, shuffle=True, random_state=self.random_state.value)
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
                self.task_type,
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

        empirical = {}
        for r in results:
            for k, score in r['raw_results'].items():
                if k not in empirical:
                    empirical[k] = []
                empirical[k].extend(score)

        with self.output() as emitter:
            emitter.emit(
                {
                    'parameter': self.get_params(),
                    'train_time': R['train_time'],
                    'test_time': R['test_time'],
                    'result': R['result'],
                    'raw_results': empirical
                }
            )


class BagParameterGridTask(Task):
    out_dir = Parameter('./eval/')

    def __init__(self, clf_type, paramGrid,
                 graph_count, scores, opt_scores,
                 subset_index=None,
                 category=None,
                 task_type=None,
                 kernel='linear'):
        self.clf_type = clf_type
        self.paramGrid = paramGrid
        self.kernel = kernel
        self.graph_count = graph_count
        self.scores = scores
        self.opt_scores = opt_scores
        self.subset_index = subset_index
        self.category = category
        self.task_type = task_type

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
                    self.task_type,
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
    random_state = Parameter(0)

    def __init__(self, clf_type, paramGrid,
                 graph_count, scores, opt_scores,
                 subset_index=None,
                 category=None,
                 task_type=None,
                 kernel='linear'):
        self.clf_type = clf_type
        self.paramGrid = paramGrid
        self.kernel = kernel
        self.graph_count = graph_count
        self.scores = scores
        self.opt_scores = opt_scores
        self.subset_index = subset_index
        self.task_type = task_type
        self.category = category

    def _index(self):
        if self.subset_index is None:
            return [x for x in range(self.graph_count)]
        else:
            return self.subset_index

    def require(self):
        index = np.array(self._index())
        loo = KFold(self.k.value, shuffle=True, random_state=self.random_state.value)
        return [
            BagParameterGridTask(
                self.clf_type,
                self.paramGrid,
                self.graph_count,
                self.scores,
                self.opt_scores,
                subset_index=train_index.tolist(),
                category=self.category,
                task_type=self.task_type,
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

        def __init__(self, h, D, category=None,
                     task_type=None, kernel='linear'):
            self.h = h
            self.D = D
            self.category = category
            self.task_type = task_type
            self.kernel = kernel

        def require(self):
            h = [h for h in range(self.h+1)]
            return [BagNormalizeGramTask(h, self.D, self.category,
                                         self.task_type,
                                         self.kernel)]

        def __taskid__(self):
            cat = 'all'
            if self.category is not None:
                cat = '_'.join(enumerateable(self.category))

            tt = ''
            if self.task_type is not None:
                tt = '_'+str(self.task_type)

            return 'BagMDSTask_%d_%d_%s_%s' % (self.h, self.D, self.kernel,
                                               cat
                                               )\
                + tt

        def output(self):
            path = self.out_dir.value + self.__taskid__() + '.json'
            return CachedTarget(
                LocalTarget(path, service=JsonService)
            )

        def run(self):
            with self.input()[0] as i:
                D = i.query()
                graphIndex = D['graphIndex']
                X = np.array(D['data'])
                del D

            dis = np.ones(X.shape, dtype=X.dtype) - X

            mds = MDS(n_components=2, dissimilarity="precomputed", n_init=10)
            X_r = mds.fit_transform(dis)
            stress = mds.stress_

            with self.output() as o:
                o.emit({
                    'graphIndex': graphIndex,
                    'data': X_r.tolist(),
                    'stress': stress
                })


class BagMDSVisualizeLabelTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, task_type=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.task_type = task_type
        self.kernel = kernel

    def require(self):
        return [BagLoadingTask(self.h, self.D),
                BagMDSTask(self.h, self.D, self.category,
                           self.task_type,
                           self.kernel)]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = '_'.join(enumerateable(self.category))

        tt = ''
        if self.task_type is not None:
            tt = '_'+str(self.task_type)

        return 'BagMDSVisualizeLabelTask_%d_%d_%s_%s' % (self.h, self.D, self.kernel,
                                                           cat
                                                           )\
            + tt

    def output(self):
        return FileTarget(self.out_dir.value + self.__taskid__() + '.png')

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
            stress = D['stress']
            del D

        y, times = self._build_maps()
        y = BagMDSVisualizeLabelTask._index_map(graphIndex, y)
        y_solve, y_time = BagMDSVisualizeLabelTask._state_time_split(y)

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

        plt.figure(1)
        plt.suptitle('MDS of GRAM dataset (h: %s, D: %s) [%s points] (Stress: %2.2f)' %
                     (str(self.h), str(self.D), str(len(X)), stress))

        plt.subplot(121)
        for color, i, t in zip(colors, range(len(aName)), aName):
            plt.scatter(X[l_solve == i, 0], X[l_solve == i, 1],
                        color='none', alpha=.8,
                        lw=2,
                        label=t,
                        edgecolors=color)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

        plt.subplot(122)
        for color, i, t in zip(colors, range(len(aName)), aName):
            plt.scatter(X[l_time == i, 0], X[l_time == i, 1],
                        color='none', alpha=.8,
                        lw=2,
                        label=t,
                        edgecolors=color)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

        path = self.output().path

        directory = os.path.dirname(path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(path)
        plt.close()


class BagMDSCategoryTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, h, D, category=None, kernel='linear'):
        self.h = h
        self.D = D
        self.category = category
        self.kernel = kernel

    def require(self):
        return [BagLoadingTask(self.h, self.D),
                BagMDSTask(self.h, self.D, self.category, self.kernel)]

    def __taskid__(self):
        cat = 'all'
        if self.category is not None:
            cat = '_'.join(enumerateable(self.category))

        return 'BagMDSCategoryTask_%d_%d_%s_%s' % (self.h, self.D, self.kernel,
                                                           cat
                                                           )

    def output(self):
        return FileTarget(self.out_dir.value + self.__taskid__() + '.png')

    def run(self):
        with self.input()[0] as i:
            bag = ProgramBags(content=i.query())

        with self.input()[1] as i:
            D = i.query()
            graphIndex = D['graphIndex']
            X = np.array(D['data'])
            stress = D['stress']
            del D

        path = self.output().path

        directory = os.path.dirname(path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        spath = os.path.splitext(path)
        spath = spath[0] + '_%s' + spath[1]

        colors = ['grey', 'red']
        aName = ['OTHER', 'cat']

        for k, V in bag.categories.items():

            l_vec = np.zeros(len(X))
            nnz = 0

            aName[1] = k

            for p in V:
                pos = graphIndex[p]
                l_vec[pos] = 1
                nnz += 1

            plt.figure(1)
            plt.subtitle('MDS of Category %s (h: %s, D: %s) [%s points] (Stress: %2.2f)' %
                         (k, str(self.h), str(self.D), str(nnz), stress))

            for color, i, t in zip(colors, range(len(aName)), aName):
                plt.scatter(X[l_vec == i, 0], X[l_vec == i, 1],
                            color='none', alpha=.8,
                            lw=2,
                            label=t,
                            edgecolors=color)
            plt.legend(loc='best', shadow=False, scatterpoints=1)

            plt.savefig(spath % k)
            plt.close()
