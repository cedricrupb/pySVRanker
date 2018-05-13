from pyTasks import task
from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash, TaskProgressHelper
from pyTasks.target import CachedTarget, LocalTarget, NetworkXService, ManagedTarget
from pyTasks.target import FileTarget, JsonService
from .graph_tasks import GraphPruningTask
from .ranking_task import ExtractInfoTask
import networkx as nx
import numpy as np
from .prepare_tasks import GraphIndexTask
import os
from os.path import dirname
from .ranking_task import DefineClassTask
from .kernel_function import select_kernel
import csv
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from pyTasks.utils import tick
from scipy.sparse import coo_matrix, diags
try:
    import mmh3
except ImportError:
    import pymmh3 as mmh3


def parseList(s):
    if not isinstance(s, str):
        return s
    state = 0
    tmp = ''
    out = []
    for i in range(len(s)):
        c = s[i]
        if state == 0:
            if c == '[':
                state = 1
        elif state == 1:
            if c == ',':
                out.append(tmp)
                tmp = ''
            elif c == "\'":
                state = 2
            elif c == ']':
                return out
        elif state == 2:
            if c == '\'':
                state = 1
            else:
                tmp += c
    raise ValueError(str(s)+" is malformed")


def indexMap(key, mapping):
    counter = 0
    if 'counter' in mapping:
        counter = mapping['counter']

    if key not in mapping:
        mapping[key] = counter
        mapping['counter'] = counter + 1

    return mapping[key]


class PrepareKernelTask(Task):
    out_dir = Parameter('./gram/')
    timeout = Parameter(None)
    rainbow = Parameter(False)
    descriptive = Parameter(False)

    def __init__(self, graph, h, D):
        self.graph = graph
        self.h = h
        self.D = D

    def require(self):
        if self.h <= 0:
            return GraphPruningTask(self.graph, self.D)
        else:
            return PrepareKernelTask(self.graph, self.h - 1, self.D)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.pickle'
        return CachedTarget(
            LocalTarget(path, service=NetworkXService)
        )

    def __taskid__(self):
        return 'PrepareKernelTask_%s_%d_%d' % (self.graph,
                                               self.h, self.D)

    def __repr__(self):
        return 'PrepareKernelTask(graph: %s, h: %d, D: %d)' %\
                (self.graph, self.h, self.D)

    def _hash(self, obj):
        hash = mmh3.hash(obj)
        if self.rainbow.value:
            if not hasattr(self, 'rainbow_table'):
                self.rainbow_table = {}
            self.rainbow_table[hash] = obj
        return hash

    def _describe(self, source, neighbours):
        prefix = ''
        if self.descriptive.value:
            prefix = str(source)[:min(8, len(source))]

        neighbours.append(source)

        return prefix + str(self._hash(
                                '_'.join(
                                    [str(t) for t in neighbours
                                     ]
                                )
                            ))

    def _collect_labels(self, graph):
        ret = {}
        for u, v, d in graph.in_edges(data=True):

            tick(self)

            source = graph.node[u]['label']
            edge_t = d['type']
            truth = d['truth']

            long_edge_label = self._describe(
                source, [edge_t, truth]
            )

            if v not in ret:
                ret[v] = []

            ret[v].append(long_edge_label)
        return ret

    def run(self):
        with self.input()[0] as graphInput:
            graph = graphInput.query()

        count = {}

        if self.h > 0:
            # Step 1
            M = self._collect_labels(graph)

            # Step 2, 3, 4
            for n, d in graph.nodes(data=True):

                tick(self)

                if n not in M:
                    continue
                label = self._describe(d['label'], sorted(M[n]))

                if label not in count:
                    count[label] = 0
                count[label] += 1

                d['label'] = label
        else:
            for n, d in graph.nodes(data=True):

                tick(self)

                label = d['label']

                if label not in count:
                    count[label] = 0
                count[label] += 1

        graph.graph['label_count'] = count

        if hasattr(self, 'rainbow_table'):
            graph.graph['rainbow_table'] = self.rainbow_table

        with self.output() as out_dirput:
            out_dirput.emit(graph)


class WLCollectorTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, graphs, h, D):
        self.graphs = graphs
        self.h = h
        self.D = D

    def require(self):
        task = []
        if self.h > 0:
            task.append(
                WLCollectorTask(self.graphs, self.h - 1, self.D))
        for g in self.graphs:
            task.append(Optional(PrepareKernelTask(g, self.h, self.D)))
        return task

    def output(self):
        return ManagedTarget(self)

    def __taskid__(self):
        return "WLCollector_%d_%d_%s" %\
                    (self.h, self.D,
                     str(containerHash(self.graphs, large=True)))

    def __repr__(self):
        return 'WLCollector(h: %d, D: %d)' % (self.h, self.D)

    def run(self):
        M = {}
        s = 0
        if self.h > 0:
            with self.input()[0] as i:
                M = i.query()
            s = 1

        paths = {}

        for i in range(s, len(self.input())):
            inputDep = self.input()[i]
            g = self.graphs[i - s]
            if inputDep is None:
                if g in M:
                    del M[g]
            else:
                paths[g] = inputDep

        for g, p in paths.items():
            if g not in M:
                M[g] = {}
            if self.h not in M[g]:
                M[g][self.h] = {}
            with p as pin:
                G = pin.query()
            count = G.graph['label_count']
            del G
            for n, c in count.items():
                M[g][self.h][n] = c

        with self.output() as o:
            o.emit(M)


class WLKernelTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, graphs, h, D):
        self.graphs = graphs
        self.h = h
        self.D = D

    def require(self):
        return WLCollectorTask(self.graphs, self.h, self.D)

    def output(self):
        return ManagedTarget(self)

    def __taskid__(self):
        return "WLKernel_%d_%d_%s" %\
                    (self.h, self.D,
                     str(containerHash(self.graphs, large=True)))

    def __repr__(self):
        return 'WLKernel(h: %d, D: %d)' % (self.h, self.D)

    def run(self):
        with self.input()[0] as i:
            M = i.query()

        graphIndex = {}
        nodeIndex = {}
        K = [None] * (self.h + 1)

        for g, D in M.items():
            gI = indexMap(g, graphIndex)
            for h, N in D.items():

                row = []
                column = []
                data = []

                for n, c in N.items():
                    nI = indexMap(n, nodeIndex)
                    row.append(gI)
                    column.append(nI)
                    data.append(c)

                K[h] = {
                    'row': row,
                    'column': column,
                    'data': data
                }
        del M

        GR = None

        for h, D in enumerate(K):
            phi = coo_matrix((D['data'], (D['row'], D['column'])),
                             shape=(graphIndex['counter'], nodeIndex['counter']),
                             dtype=np.uint64).tocsr()
            del K[h]
            T = phi.dot(phi.transpose())
            if GR is None:
                GR = T
            else:
                GR += T
        del K

        with self.output() as o:
            o.emit((graphIndex, GR))


class CustomKernelTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, kernel_type, graphs, h, D):
        self.kernel_type = kernel_type
        self.graphs = graphs
        self.h = h
        self.D = D

    def require(self):
        return WLCollectorTask(self.graphs, self.h, self.D)

    def output(self):
        return ManagedTarget(self)

    def __taskid__(self):
        return "CustomKernelTask(%s)_%d_%d_%s" %\
                    (self.kernel_type, self.h, self.D,
                     str(containerHash(self.graphs, large=True)))

    def __repr__(self):
        return 'CustomKernel[%s](h: %d, D: %d)' % (self.kernel_type, self.h,
                                                   self.D)

    @staticmethod
    def pairwise_index(D1, D2):
        index = {}
        O1 = {}

        for d, v in D1.items():
            O1[indexMap(d, index)] = v

        O2 = {}

        for d, v in D2.items():
            O2[indexMap(d, index)] = v

        V1 = np.zeros((index['counter']), dtype=np.int64)

        for o, v in O1.items():
            V1[o] = v

        V2 = np.zeros((index['counter']), dtype=np.int64)

        for o, v in O2.items():
            V2[o] = v

        return V1, V2

    def pairwise_kernel(self, X, Y):
        VX, VY = CustomKernelTask.pairwise_index(X, Y)

        return self._kernel(VX, VY)

    @staticmethod
    def dis_to_sim(X):
        MAX = np.full(X.shape, np.amax(X), dtype=np.float64)

        return MAX - X

    def run(self):
        self._kernel = select_kernel(self.kernel_type)

        with self.input()[0] as i:
            M = i.query()

        graphIndex = {}
        K = [None] * (self.h + 1)

        for g, D in M.items():
            gI = indexMap(g, graphIndex)
            for h, N in D.items():
                if K[h] is None:
                    K[h] = {}
                K[h][gI] = N
        del M

        GR = None

        for h, D in enumerate(K):
            T_GR = np.zeros((graphIndex['counter'], graphIndex['counter']),
                            dtype=np.float64)

            for i in range(graphIndex['counter']):
                for j in range(graphIndex['counter']):
                    if i <= j:
                        T_GR[i, j] = self.pairwise_kernel(D[i], D[j])
                        T_GR[j, i] = T_GR[i, j]

            if T_GR[0, 0] == 0:
                T_GR = CustomKernelTask.dis_to_sim(T_GR)

            if GR is None:
                GR = T_GR
            else:
                GR += T_GR
            del K[h]
        del K

        with self.output() as o:
            o.emit((graphIndex, GR))


class NormalizedWLKernelTask(Task):
    out_dir = Parameter('./gram/')
    custom_kernel = Parameter(None)

    def __init__(self, graphs, h, D):
        self.graphs = graphs
        self.h = h
        self.D = D

    def require(self):
        if self.custom_kernel.value is None:
            return WLKernelTask(self.graphs, self.h, self.D)
        else:
            return CustomKernelTask(self.custom_kernel.value, self.graphs,
                                    self.h, self.D)

    def output(self):
        return ManagedTarget(self)

    def __taskid__(self):
        return "NormWLKernel_%d_%d_%s" %\
                    (self.h, self.D,
                     str(containerHash(self.graphs, large=True)))

    def __repr__(self):
        return 'NormalizedKernel(h: %d, D: %d)' % (self.h, self.D)

    def run(self):
        with self.input()[0] as i:
            graphIndex, GR = i.query()

        D = diags(1/np.sqrt(GR.diagonal()))

        GR = D * GR * D

        with self.output() as o:
            o.emit((graphIndex, GR))


class ExtractKernelBagTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, graph, h, D):
        self.graph = graph
        self.h = h
        self.D = D

    def require(self):
        return PrepareKernelTask(self.graph, self.h, self.D)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'ExtractKernelBagTask_%s_%d_%d' % (self.graph,
                                                  self.h, self.D)

    def __repr__(self):
        return 'ExtractKernelBagTask(graph: %s, h: %d, D: %d)' %\
                (self.graph, self.h, self.D)

    def run(self):
        with self.input()[0] as i:
            G = i.query()

        with self.output() as o:
            o.emit(G.graph['label_count'])


class ExtractKernelEntitiesTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, graphs, h, D):
        self.graphs = graphs
        self.h = h
        self.D = D

    def require(self):
        out = [ExtractInfoTask(self.graphs), GraphIndexTask()]

        for g in self.graphs:
            out.append(Optional(ExtractKernelBagTask(g, self.h, self.D)))

        return out

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'ExtractKernelEntitiesTask_%d_%d_%s' % (self.h, self.D,
                                                       str(
                                                        containerHash(self.graphs)
                                                       ))

    def run(self):
        with self.input()[0] as i:
            info = i.query()

        with self.input()[1] as i:
            index = i.query()['index']

        out_dict = {}

        for i, g in enumerate(self.graphs):
            name = index[g]

            if self.input()[i + 2] is None:
                continue

            with self.input()[i + 2] as bag_input:
                bag = bag_input.query()

            out_dict[g] = {
                'file': name,
                'kernel_bag': bag,
                'label': info[g]
            }

        with self.output() as out_file:
            out_file.emit(out_dict)


class MDSTask(Task):
    out_dir = Parameter('./gram/')

    def __init__(self, graphs, h, D):
        self.graphs = graphs
        self.h = h
        self.D = D

    def require(self):
        return [
            NormalizedWLKernelTask(self.graphs, self.h, self.D),
            DefineClassTask(self.graphs)
            ]

    def output(self):
        return FileTarget(self.out_dir.value+self.__taskid__()+'.png')

    def __taskid__(self):
        return "MDS_%d_%d_%s" %\
                    (self.h, self.D,
                     str(containerHash(self.graphs, large=True)))

    def __repr__(self):
        return 'MDS(h: %d, D: %d)' % (self.h, self.D)

    def __evalScore(self, score):
        out = None

        for k, v in score.items():
            if v == 'correct':
                if out is None:
                    out = k
                else:
                    out = 'UNKNOWN'

        return out

    def __evalTime(self, t_rank):
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

    def run(self):
        with self.input()[0] as i:
            graphIndex, GR = i.query()

        with self.input()[1] as i:
            R = i.query()

        dis = np.ones(GR.shape, dtype=GR.dtype) - GR
        colors = ['grey', 'green', 'red']
        tName = ['UNKNOWN', 'IUV', 'ESBMC']
        aName = ['UNKNOWN', 'Tester', 'Verificator']

        lScore = np.zeros(len(GR))
        lTime = np.zeros(len(GR))
        for index, D in R.items():
            if index not in graphIndex:
                continue
            gI = graphIndex[index]
            score = self.__evalScore(D['score'])
            time = self.__evalTime(D['time_rank'])
            for i, t in enumerate(tName):
                if score == t:
                    lScore[gI] = i
                if time == t:
                    lTime[gI] = i

        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=10)
        X_r = mds.fit_transform(dis)
        stress = mds.stress_

        plt.figure(1)
        plt.suptitle('MDS of GRAM dataset (h: %s, D: %s) [%s points] (Stress: %2.2f)' %
                     (str(self.h), str(self.D), str(len(X_r)), stress))

        plt.subplot(121)
        for color, i, t in zip(colors, range(len(aName)), aName):
            plt.scatter(X_r[lScore == i, 0], X_r[lScore == i, 1],
                        color=color, alpha=.8,
                        lw=2,
                        label=t)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

        plt.subplot(122)
        for color, i, t in zip(colors, range(len(aName)), aName):
            plt.scatter(X_r[lTime == i, 0], X_r[lTime == i, 1],
                        color=color, alpha=.8,
                        lw=2,
                        label=t)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

        path = self.output().path

        directory = dirname(path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    config = {
        "GraphSubTask": {
            "graphPath": "/Users/cedricrichter/Documents/Arbeit/Ranking/PyPRSVT/static/results-tb-raw/",
            "out_dir": "./test/",
            "cpaChecker": "/Users/cedricrichter/Documents/Arbeit/Ranking/cpachecker"
                },
        "GraphConvertTask": {
            "out_dir": "./test/"
        },
        "CategoryLookupTask": {
            "graphPaths": "/Users/cedricrichter/Documents/Arbeit/Ranking/PyPRSVT/static/results-tb-raw/"
        },
        "MemcachedTarget": {
            "baseDir": "./cache/"
        },
        "GraphIndexTask": {
            "categories": ['array-examples',
                           'array-industry-pattern',
                           "bitvector-loops",
                           "bitvector-regression",
                           "bitvector"]
        },
        "GraphPruningTask": {
            "out_dir": "./test/"
        },
        "RemoteTarget": {
            "host": "pc-wehr-serv1.cs.upb.de",
            "user": "cedricr",
            "password": "wehrheim01",
            "remote_path": "./test/"
        }
            }

    injector = task.ParameterInjector(config)
    planner = task.TaskPlanner(injector=injector)
    exe = task.TaskExecutor()

    task = GraphIndexTask()
    plan = planner.plan(task)
    helper = TaskProgressHelper(plan)
    exe.executePlan(plan)

    with helper.output(task) as js:
        index = js.query()

    graphs = []

    for k, v in index['categories'].items():
        graphs.extend(v)

    task = MDSTask(graphs, 2, 5)
    plan = planner.plan(task, graph=plan)
    exe.executePlan(plan)
