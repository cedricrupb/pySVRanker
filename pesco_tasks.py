from pyTasks.task import Task, Parameter, Optional
from pyTasks.target import LocalTarget, JsonService, FileTarget
import os
import subprocess
from subprocess import PIPE
from os.path import join, isdir, isfile
import logging
import shutil
import json
from scipy.sparse import coo_matrix, diags, issparse
import numpy as np
from .kernel_function import select_full
from tqdm import trange, tqdm
from .kernel_function import is_pairwise, is_absolute

from .scripts import index_svcomp, generate_bag
from .bag_tasks import is_correct, is_false, is_faster, index
from .bag import normalize_gram


def checkIfEmpty(path):
    if not isfile(path):
        return True

    return os.stat(path).st_size == 0


class PescoGraphTask(Task):
    out_dir = Parameter("./pesco/")
    pesco_path = Parameter('')
    svcomp_path = Parameter('')
    timeout = Parameter(None)
    heap = Parameter("15000m")

    def __init__(self, input_path):
        self.input_path = input_path

    def _graphId(self, s):
        s = s.replace(self.svcomp_path.value, "")
        s = s.replace("/", "_")
        s = s.replace(".", "_")
        return s

    def require(self):
        pass

    def __taskid__(self):
        return "PescoGraphTask_%s" % self._graphId(
            self.input_path
        )

    def output(self):
        if not hasattr(self, "output_path"):
            self.output_path =\
                join(self.pesco_path.value,
                     "output",
                     self._graphId(self.input_path)+".dfs")
        return FileTarget(
            self.output_path
        )

    def run(self):
        pesco_path = self.pesco_path.value
        path_to_source = self.input_path

        if self.svcomp_path.value not in path_to_source:
            path_to_source = join(self.svcomp_path.value, path_to_source)

        run_path = join(pesco_path, "scripts", "cpa.sh")
        output_path = self.output().path

        if not isdir(pesco_path):
            raise ValueError("Unknown pesco path %s" % pesco_path)
        if not (isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
            raise ValueError('path_to_source is no valid filepath. [%s]' % path_to_source)

        proc = subprocess.run(
                                [run_path,
                                 "-graphgen",
                                 "-heap", self.heap.value,
                                 "-Xss512k",
                                 "-setprop", "graphGen.output="+output_path,
                                 path_to_source
                                 ],
                                check=False, stdout=PIPE, stderr=PIPE,
                                timeout=self.timeout.value
                                )

        if proc.returncode != 0 or checkIfEmpty(output_path):
            logging.error(proc.args)
            logging.error(proc.stdout.decode('utf-8'))
            logging.error(proc.stderr.decode('utf-8'))
            raise ValueError("Something went wrong while processing: %s" % path_to_source)

        with open(output_path, "r") as i:
            with self.output() as o:
                shutil.copyfileobj(i, o)


class PescoGraphIndex(Task):
    out_dir = Parameter("./pesco/")

    def __init__(self, paths):
        self.paths = paths

    def require(self):
        return [
            Optional(
                PescoGraphTask(graph)
            )
            for graph in self.paths
        ]

    def __taskid__(self):
        return "PescoGraphIndex_%d" % len(self.paths)

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService)

    def run(self):
        index = {'counter': 0}

        for i, inp in enumerate(self.input()):
            if inp is not None:
                id = inp.path
                id = id.replace("output/", "")
                id = id.replace("PescoGraphTask_", "")
                id = id.replace(".dfs", "")
                if id not in index:
                    index[id] = index['counter']
                    index['counter'] += 1

        with self.output() as o:
            o.emit(index)


class DatasetLabelTask(Task):
    out_dir = Parameter("./dataset/")

    def __init__(self, path, svcomp_name, directory, csv=False):
        self.path = path
        self.svcomp_name = svcomp_name
        self.directory = directory
        self.csv = csv

    def require(self):
        return [
            PescoGraphTask(self.path),
            SVCompRanking(self.svcomp_name, self.directory,
                          self.csv)
        ]

    def __taskid__(self):
        return "DatasetLabelTask_%s_%s" % (self.svcomp_name, self.path.replace("/", "_").replace(".", "_"))

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):
        with self.input()[1] as inp:
            rankings = inp.query()

        ranking = []
        if self.path in rankings:
            ranking = rankings[self.path]
        else:
            raise ValueError("Unknown path %s" % self.path)

        with self.input[0] as inp:
            G = json.load(inp)

        with self.output() as o:
            o.emit({
                'file': self.path,
                'svcomp': self.svcomp_name,
                'rankings': ranking,
                'graph': G
            })


class DatasetCreationTask(Task):
    out_dir = Parameter("./dataset/")

    def __init__(self, paths, svcomp_name, directory, csv=False):
        self.paths = paths
        self.svcomp_name = svcomp_name
        self.directory = directory
        self.csv = csv

    def require(self):
        return [
            Optional(
                DatasetLabelTask(p, self.svcomp_name, self.directory, self.csv)
            )
            for p in self.paths
        ]

    def __taskid__(self):
        return "DatasetCreationTask_%s" % (self.svcomp_name)

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):

        index = {'counter': 0}

        for i, inp in enumerate(self.input()):
            if inp is not None:
                p = self.paths[i]
                if p not in index:
                    index[p] = index['counter']
                    index['counter'] += 1

        with self.output() as o:
            o.emit(index)


class SVCompIndex(Task):
    out_dir = Parameter("./index/")
    bench_prefix = Parameter("../sv-benchmarks/c/")

    def __init__(self, svcomp_name, directory, csv=False):
        self.directory = directory
        self.svcomp_name = svcomp_name
        self.csv = csv

    def require(self):
        pass

    def __taskid__(self):
        return "SVCompIndex_%s" % self.svcomp_name

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):

        results = index_svcomp.parse(self.directory, self.bench_prefix.value,
                                     self.csv)

        with self.output() as o:
            o.emit(results)


def _is_list(obj):
    if isinstance(obj, str):
        return False
    try:
        _ = (e for e in obj)
        return True
    except TypeError:
        return False


def rank_compare(l1, l2):
    if l1['solve'] > l2['solve']:
        return 1
    if l2['solve'] > l1['solve']:
        return 0
    if l1['time'] >= 900 and l2['time'] >= 900:
        return 0.5
    return 1 if l1['time'] < l2['time'] else 0


def rank_tools(L, tools):
    ranks = {}
    n = len(tools)

    for k, V in L.items():
        ranks[k] = {}
        for prop, V in V.items():

            for t in tools:
                if t not in V:
                    V[t] = {'solve': 0, 'time': 900}

            if prop not in ranks[k]:
                ranks[k][prop] = {}
            for i in range(n-1):
                for j in range(i+1, n):
                    t1 = tools[i]
                    t2 = tools[j]
                    l1 = V[t1]
                    l2 = V[t2]

                    if t1 not in ranks[k][prop]:
                        ranks[k][prop][t1] = 0
                    if t2 not in ranks[k][prop]:
                        ranks[k][prop][t2] = 0

                    c = rank_compare(l1, l2)
                    ranks[k][prop][t1] += c
                    ranks[k][prop][t2] += (1 - c)

    for k, V in ranks.items():
        for prop, V in list(V.items()):
            if len(V) == 0:
                continue
            value = sorted(list(V.items()), key=lambda X: X[1], reverse=True)
            R = [value[0][0]]
            for i in range(1, len(value)):
                t, v = value[i]
                lr = R[-1]
                if _is_list(lr):
                    if V[lr[-1]] == v:
                        lr.append(t)
                    else:
                        R.append(t)
                else:
                    if V[lr] == v:
                        R[-1] = [lr, t]
                    else:
                        R.append(t)
            ranks[k][prop] = R

    return ranks


class SVCompRanking(Task):
    out_dir = Parameter("./index/")

    def __init__(self, svcomp_name, directory, csv=False):
        self.directory = directory
        self.svcomp_name = svcomp_name
        self.csv = csv

    def require(self):
        return SVCompIndex(
            self.svcomp_name, self.directory,
            self.csv
        )

    def __taskid__(self):
        return "SVCompRanking_%s" % self.svcomp_name

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    @staticmethod
    def common_tools(L):
        tools = {}
        counter = 0

        for k, V in L.items():
            for prop, V in V.items():
                counter += 1
                for tool in V.keys():
                    if tool == 'name':
                        continue
                    if tool not in tools:
                        tools[tool] = 0
                    tools[tool] += 1
        result = []
        for tool, c in tools.items():
            if c >= counter-5 and 'cpa-bam' not in tool:
                result.append(tool)
        return result

    def run(self):
        with self.input()[0] as i:
            L = i.query()

        tools = SVCompRanking.common_tools(L)

        ranks = rank_tools(L, tools)

        with self.output() as o:
            o.emit(ranks)


class SVCompGraphIndexTask(Task):
    out_dir = Parameter("./index/")

    def __init__(self, svcomp_name, directory, csv=False):
        self.directory = directory
        self.svcomp_name = svcomp_name
        self.csv = csv

    def require(self):
        return SVCompIndex(
            self.svcomp_name, self.directory,
            self.csv
        )

    def __taskid__(self):
        return "SVCompGraphIndexTask_%s" % self.svcomp_name

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):
        with self.input()[0] as i:
            L = i.query()

        graphIndex = {'counter': 0}
        categories = {}

        for k, V in L.items():
            if k not in graphIndex:
                graphIndex[k] = graphIndex['counter']
                graphIndex['counter'] += 1
            for prop in V.keys():
                if prop not in categories:
                    categories[prop] = []
                categories[prop].append(graphIndex[k])

        with self.output() as o:
            o.emit({
                'index': graphIndex,
                'categories': categories
            })


class PescoWLTransformerTask(Task):
    out_dir = Parameter("./wlj/")

    def __init__(self, path, max_i, max_D):
        self.path = path
        self.max_i = max_i
        self.max_D = max_D

    def require(self):
        return PescoGraphTask(self.path)

    def __taskid__(self):
        return "PescoWLTransformerTask_%s_%d_%d" % (self.path.replace("/", "_").replace(".", "_"), self.max_i, self.max_D)

    def output(self):
        return LocalTarget(
            self.out_dir.value + self.__taskid__() + ".json", service=JsonService
        )

    def run(self):

        with self.input()[0] as i:
            G = generate_bag.parse_dfs_nx(
                json.load(i)
            )

        stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.size(),
            "max_indegree": max(G.in_degree()),
            "max_outdegree": max(G.out_degree())
        }

        bags = {}

        for d in range(self.max_D, 0, -1):
            generate_bag.truncate(G, d)

            relabel = {}
            for i in range(self.max_i + 1):

                if i not in bags:
                    bags[i] = {}

                if d not in bags[i]:
                    bags[i][d] = {}

                bag = generate_bag.labelCount(G, relabel)

                bags[i][d] = bag

                relabel = generate_bag.wlGraphRelabel(G, relabel)

        with self.output() as o:
            o.emit({
                'statistics': stats,
                'kernel_bag': bags
            })


class PescoWLStatisticsTask(Task):
    out_dir = Parameter("./stats/")

    def __init__(self, paths, max_i, max_D,
                 svcomp_name, directory, csv=False):
        self.svcomp_name = svcomp_name
        self.paths = paths
        self.max_i = max_i
        self.max_D = max_D
        self.directory = directory
        self.csv = csv

    def require(self):
        base = [
                SVCompGraphIndexTask(self.svcomp_name,
                                     self.directory,
                                     self.csv)
                ]
        base.extend([
            PescoWLTransformerTask(
                p, self.max_i, self.max_D
            )
            for p in self.paths
        ])
        return base

    def __taskid__(self):
        return "PescoWLStatisticsTask_%s" % (self.svcomp_name)

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):
        with self.input()[0] as i:
            local_index = i.query()

        cat_index = {}

        statistics = {'overall': {}}

        for k, C in local_index['categories'].items():
            statistics[k] = {}
            for c in C:
                if c not in cat_index:
                    cat_index[c] = ['overall']
                cat_index[c].append(k)

        for i in trange(1, len(self.input())):
            with self.input()[i] as j:
                stats = j.query()['statistics']
            p = self.paths[i - 1]
            pi = local_index['index'][p]
            props = cat_index[pi]

            for s, c in stats.items():
                for prop in props:
                    if s not in statistics[prop]:
                        statistics[prop][s] = []
                    if isinstance(c, list):
                        c = c[0]
                    statistics[prop][s].append(c)

        for prop, V in statistics.items():
            for s in list(V.keys()):
                V[s] = {
                    'mean': float(np.mean(V[s])),
                    'std': float(np.std(V[s])),
                    'min': float(np.min(V[s])),
                    'max': float(np.max(V[s])),
                    'len': int(len(V[s]))
                }

        with self.output() as o:
            o.emit(statistics)


def pairwise_kernel(kernel, X, Y):
    """Generate a kernel only between two tasks."""
    return kernel(X, Y)


def dis_to_sim(X):
    """Return a similarity measure by using a distance measure."""
    MAX = np.full(X.shape, np.amax(X), dtype=np.float64)

    return MAX - X


def _pairwise_gram(kernel, feature_matrix):
    gC = feature_matrix.shape[0]
    T_GR = np.zeros((gC, gC),
                    dtype=np.float64)

    for i in trange(gC):
        for j in range(i+1, gC):
            if i <= j:
                X = feature_matrix[i, :]
                Y = feature_matrix[j, :]
                T_GR[i, j] = pairwise_kernel(
                    kernel, X.transpose(), Y.transpose()
                )

                T_GR[j, i] = T_GR[i, j]

    if T_GR[0, 0] == 0:
        T_GR = dis_to_sim(T_GR)

    return T_GR


def _custom_gram(kernel, feature_matrix):
    if is_pairwise(kernel):
        return _pairwise_gram(kernel, feature_matrix)
    elif is_absolute(kernel):
        return kernel(feature_matrix)
    else:
        raise ValueError('Kernel has to accept 1 (complete feature set)' +
                         ' or 2 (pairwise 1-D) matrices')


class PescoGramTask(Task):
    out_dir = Parameter("./gram/")

    def __init__(self, paths, i, d, max_i, max_D,
                 svcomp_name, directory, csv=False,
                 kernel='linear'):
        self.svcomp_name = svcomp_name
        self.paths = paths
        self.i = i
        self.d = d
        self.max_i = max_i
        self.max_D = max_D
        self.directory = directory
        self.csv = csv
        self.kernel = kernel

    def require(self):
        base = [
                SVCompGraphIndexTask(self.svcomp_name,
                                     self.directory,
                                     self.csv)
                ]
        base.extend([
            PescoWLTransformerTask(
                p, self.max_i, self.max_D
            )
            for p in self.paths
        ])
        return base

    def __taskid__(self):
        return "PescoGramTask_%s_%s_%d_%d" % (self.svcomp_name, self.kernel, self.i, self.d)

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def _feature_matrix(self, graph_index):
        index_path = {k: i for i, k in enumerate(self.paths)}

        gC = graph_index['counter']
        del graph_index['counter']

        featureIndex = {'counter': 0}

        row = []
        column = []
        data = []

        for graph_name, gI in tqdm(list(graph_index.items())):
            p_index = index_path[graph_name]
            inp = self.input()[p_index+1]
            with inp as i:
                K = i.query()
                K = K['kernel_bag']
                K = K[str(self.i)][str(self.d)]
            for n, c in K.items():
                if n not in featureIndex:
                    featureIndex[n] = featureIndex['counter']
                    featureIndex['counter'] += 1
                fI = featureIndex[n]
                row.append(gI)
                column.append(fI)
                data.append(c)

        return coo_matrix(
            (data, (row, column)),
            shape=(gC, featureIndex['counter']),
            dtype=np.uint64
        ).tocsr()

    def run(self):
        with self.input()[0] as inp:
            graph_index = inp.query()['index']

        feature_matrix = self._feature_matrix(graph_index)

        if self.kernel == 'linear':
            K = feature_matrix.dot(feature_matrix.transpose())
        else:
            kernel = select_full(self.kernel)
            if kernel is None:
                raise ValueError('Unknown kernel %s' % self.kernel)
            K = _custom_gram(kernel, feature_matrix)
            if issparse(K):
                K = K.toarray()

        print(K.shape)

        if issparse(K):
            K = K.toarray()

        data = K.tolist()

        with self.output() as o:
            o.emit(data)


class PescoSumGramTask(Task):
    out_dir = Parameter("./gram/")

    def __init__(self, paths, i, d, max_i, max_D,
                 svcomp_name, directory, csv=False,
                 kernel='linear'):
        self.svcomp_name = svcomp_name
        self.paths = paths
        self.i = i
        self.d = d
        self.max_i = max_i
        self.max_D = max_D
        self.directory = directory
        self.csv = csv
        self.kernel = kernel

    def require(self):
        return [
            PescoGramTask(
                self.paths, i, self.d, self.max_i,
                self.max_D, self.svcomp_name, self.directory,
                self.csv, self.kernel
            ) for i in range(self.i+1)
        ]

    def __taskid__(self):
        return "PescoSumGramTask_%s_%s_%d_%d" % (self.svcomp_name, self.kernel, self.i, self.d)

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):
        GR = None

        for inp in self.input():
            with inp as i:
                gram = np.array(
                    i.query()
                )
            if GR is None:
                GR = gram
            else:
                GR += gram
            del gram

        data = GR.tolist()

        with self.output() as o:
            o.emit(data)


class PescoNormGramTask(Task):
    out_dir = Parameter("./gram/")

    def __init__(self, paths, i, d, max_i, max_D,
                 svcomp_name, directory, csv=False,
                 kernel='linear'):
        self.svcomp_name = svcomp_name
        self.paths = paths
        self.i = i
        self.d = d
        self.max_i = max_i
        self.max_D = max_D
        self.directory = directory
        self.csv = csv
        self.kernel = kernel

    def require(self):
        return PescoSumGramTask(
            self.paths, self.i, self.d,
            self.max_i, self.max_D,
            self.svcomp_name, self.directory,
            self.csv, self.kernel
        )

    def __taskid__(self):
        return "PescoNormGramTask_%s_%s_%d_%d" % (self.svcomp_name, self.kernel, self.i, self.d)

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    def run(self):
        with self.input()[0] as i:
            gram = np.array(
                i.query()
            )

        data = normalize_gram(gram).tolist()

        with self.output() as o:
            o.emit(data)


def y_index(graphIndex):
    counter = 0
    label_index = {}
    rv = {v: k for k, v in graphIndex['index'].items()}

    for prop, L in graphIndex['categories'].items():
        if prop not in label_index:
            label_index[prop] = {}
        for ix in L:
            k = rv[ix]
            if k not in label_index[prop]:
                label_index[prop][k] = counter
                counter += 1

    return label_index, counter


def compare(l1, l2):
    if l1['solve'] > l2['solve']:
        return 1
    if l2['solve'] > l1['solve']:
        return -1
    if l1['time'] >= 900 and l2['time'] >= 900:
        return 0
    return 1 if l1['time'] < l2['time'] else -1


def label_clf(graphIndex, L, tools):
    y_ix, N = y_index(graphIndex)
    n = len(tools)
    w = int(n * (n-1) / 2)
    M = np.zeros((N, w))

    for k, label in L.items():
        for prop, label in label.items():

            for t in tools:
                if t not in label:
                    label[t] = {'solve': 0, 'time': 900}

            lix = y_ix[prop][k]
            for i in range(n-1):
                for j in range(i+1, n):
                    pix = index(i, j, n) - n

                    l1 = label[tools[i]]
                    l2 = label[tools[j]]
                    M[lix, pix] = compare(l1, l2)

    return y_ix, M


class SVCompLabelMatrixTask(Task):
    out_dir = Parameter("./index/")

    def __init__(self, svcomp_name, directory, csv=False):
        self.directory = directory
        self.svcomp_name = svcomp_name
        self.csv = csv

    def require(self):
        return [SVCompIndex(
            self.svcomp_name, self.directory,
            self.csv
        ), SVCompGraphIndexTask(self.svcomp_name,
                                self.directory,
                                self.csv)]

    def __taskid__(self):
        return "SVCompLabelMatrixTask_%s" % self.svcomp_name

    def output(self):
        return LocalTarget(
            self.out_dir.value+self.__taskid__()+".json", service=JsonService
        )

    @staticmethod
    def common_tools(L):
        tools = {}
        counter = 0

        for k, V in L.items():
            for prop, V in V.items():
                counter += 1
                for tool in V.keys():
                    if tool == 'name':
                        continue
                    if tool not in tools:
                        tools[tool] = 0
                    tools[tool] += 1
        result = []
        for tool, c in tools.items():
            if c >= counter-5 and 'cpa-bam' not in tool:
                result.append(tool)
        return result

    def run(self):
        with self.input()[0] as i:
            L = i.query()

        with self.input()[1] as inp:
            graph_index = inp.query()

        tools = SVCompLabelMatrixTask.common_tools(L)

        label_index, matrix = label_clf(graph_index, L, tools)

        with self.output() as o:
            o.emit({
                'tools': tools,
                'index': label_index,
                'matrix': matrix.tolist()
            })
