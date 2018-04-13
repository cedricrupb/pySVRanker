from pyTasks import task
from pyTasks.task import Task, Parameter, TargetTask, Optional
from pyTasks.task import containerHash
from pyTasks.target import CachedTarget, LocalTarget, NetworkXService, JsonService
from os.path import abspath, join, isdir, isfile
import logging
import subprocess
from subprocess import PIPE
import re
import json
from enum import Enum
import networkx as nx
from .prepare_tasks import GraphIndexTask
from pyTasks.utils import tick
import os


class EdgeType(Enum):
    de = 1
    ce = 2
    cfe = 3
    se = 4
    t = 5
    f = 6
    dummy = 7


def _read_node_labeling(labels_path, node_relabel={}):
    labels = {}
    with open(labels_path) as f:
        for line in f:
            m = re.match(r"([0-9]+),([A-Z_0-9]+)\n", line)
            if m is not None:
                node = m.group(1)
                label = m.group(2)
                if label in node_relabel:
                    label = node_relabel[label]
                labels[node] = label
    return labels


def _read_edge_labeling(labels_path):
    labels = {}
    with open(labels_path) as f:
        for line in f:
            m = re.match(r"([0-9]+),([0-9]+),([0-9]+),([0-9]+)\n", line)
            if m is not None:
                # Todo add for multigraphs
                edge = (m.group(1), m.group(2), m.group(3))
                # edge = (m.group(1), m.group(2))
                labels[edge] = m.group(4)
    return labels


def _parse_edge(edge_types):
    types = {}
    for edge, l in edge_types.items():
        edge = (edge[0], edge[1], int(edge[2]))
        types[edge] = EdgeType(int(l))
    return types


def _parse_node_depth(node_depth):
    types = {}
    for node, l in node_depth.items():
        types[node] = int(l)
    return types


class GraphTask(Task):
    out_dir = Parameter('./out/graph/')
    cpaChecker = Parameter('./cpa/')
    heap = Parameter("16384m")
    timeout = Parameter(None)
    localize = Parameter(None)
    node_relabel = Parameter({})

    def __init__(self, name):
        self.name = name

    def require(self):
        return GraphIndexTask()

    def _localize(self, path):
        if self.localize.value is not None:
            return path.replace(self.localize.value[0],
                                self.localize.value[1])
        return path

    def run_first(self):

        with self.input()[0] as i:
            index = i.query()
            index = index['index']

            if self.name not in index:
                raise ValueError('%s does not exist' % self.name)

            path_to_source = self._localize(abspath(index[self.name]))

        __path_to_cpachecker__ = self.cpaChecker.value
        cpash_path = join(__path_to_cpachecker__, 'scripts', 'cpa.sh')
        output = 'output'
        output_path = join(__path_to_cpachecker__, output)
        graph_path = join(__path_to_cpachecker__, output, 'vtask_graph.graphml')
        node_labels_path = join(__path_to_cpachecker__, output, 'nodes.labels')
        edge_types_path = join(__path_to_cpachecker__, output, 'edge_types.labels')
        edge_truth_path = join(__path_to_cpachecker__, output, 'edge_truth.labels')
        node_depths_path = join(__path_to_cpachecker__, output, 'node_depth.labels')

        if not isdir(__path_to_cpachecker__):
            raise ValueError('CPAChecker directory not found')
        if not (isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
            raise ValueError('path_to_source is no valid filepath. [%s]' % path_to_source)
        try:
            proc = subprocess.run([cpash_path,
                                   '-graphgenAnalysis',
                                   '-heap', self.heap.value,
                                   '-skipRecursion', path_to_source,
                                   '-outputpath', __path_to_cpachecker__
                                   ],
                                  check=False, stdout=PIPE, stderr=PIPE,
                                  timeout=self.timeout.value)
            match_vresult = re.search(r'Verification\sresult:\s([A-Z]+)\.', str(proc.stdout))
            if match_vresult is None:
                raise ValueError('Invalid output of CPAChecker.')
            if match_vresult.group(1) != 'TRUE':
                raise ValueError('ASTCollector Analysis failed:')
            if not isfile(graph_path):
                raise ValueError('Invalid output of CPAChecker: Missing graph output')
            if not isfile(node_labels_path):
                raise ValueError('Invalid output of CPAChecker: Missing node labels output')
            if not isfile(edge_types_path):
                raise ValueError('Invalid output of CPAChecker: Missing edge types output')
            if not isfile(edge_truth_path):
                raise ValueError('Invalid output of CPAChecker: Missing edge truth values output')
            if not isfile(node_depths_path):
                raise ValueError('Invalid output of CPAChecker: Missing node depths output')
        except ValueError as err:
            logging.error(err)
            logging.error(proc.args)
            logging.error(proc.stdout.decode('utf-8'))
            logging.error(proc.stderr.decode('utf-8'))
            raise err

        return graph_path, node_labels_path, edge_types_path,\
            edge_truth_path, node_depths_path

    def cleanup(self, graph_path, node_labels_path, edge_types_path,
                edge_truth_path, node_depths_path):
        if os.path.isfile(graph_path):
            os.remove(graph_path)

        if os.path.isfile(node_labels_path):
            os.remove(node_labels_path)

        if os.path.isfile(edge_types_path):
            os.remove(edge_types_path)

        if os.path.isfile(edge_truth_path):
            os.remove(edge_truth_path)

        if os.path.isfile(node_depths_path):
            os.remove(node_depths_path)

    def run(self):
        graph_path, node_labels_path, edge_types_path, edge_truth_path,\
            node_depths_path = self.run_first()

        tick(self)

        nx_digraph = nx.read_graphml(graph_path)

        node_labels = _read_node_labeling(node_labels_path,
                                          self.node_relabel.value)
        nx.set_node_attributes(nx_digraph, name='label', values=node_labels)

        tick(self)

        edge_types = _read_edge_labeling(edge_types_path)
        parsed_edge_types = _parse_edge(edge_types)
        nx.set_edge_attributes(nx_digraph, name='type', values=parsed_edge_types)

        tick(self)

        edge_truth = _read_edge_labeling(edge_truth_path)
        parsed_edge_truth = _parse_edge(edge_truth)
        nx.set_edge_attributes(nx_digraph, name='truth', values=parsed_edge_truth)

        tick(self)

        node_depths = _read_node_labeling(node_depths_path)
        parsed_node_depths = _parse_node_depth(node_depths)
        nx.set_node_attributes(nx_digraph, name='depth', values=parsed_node_depths)

        self.cleanup(graph_path, node_labels_path, edge_types_path,
                     edge_truth_path, node_depths_path)

        with self.output() as o:
            o.emit(nx_digraph)

    def output(self):
        path = self.out_dir.value + self.name + '.pickle'
        return CachedTarget(
            LocalTarget(path, service=NetworkXService)
        )

    def __taskid__(self):
        return 'GraphTask_' + self.name


class GraphPruningTask(Task):
    out_dir = Parameter('./out/graph/')
    allowedTypes = Parameter([1, 2, 3, 4])
    timeout = Parameter(None)

    def __get_allowed(self):
        types = self.allowedTypes.value
        if not all([EdgeType(t) in EdgeType for t in types]):
            raise ValueError('Unknown edge type detected')

        return [EdgeType(t) for t in types]

    def __init__(self, name, maxDepth):
        self.name = name
        self.maxDepth = maxDepth

    def require(self):
        return GraphTask(self.name)

    def run(self):
        with self.input()[0] as i:
            graph = i.query()

        remV = set([])
        remE = set([])

        types = self.__get_allowed()

        for n, nbrs in graph.adjacency():
            depth = graph.node[n]['depth']
            if depth > self.maxDepth:
                remV.add(n)
            for nbr, keydict in nbrs.items():
                for key, eattr in keydict.items():
                    tick(self)
                    if eattr['type'] not in types:
                        remE.add((n, nbr))

        graph.remove_nodes_from(remV)
        graph.remove_edges_from(remE)

        with self.output() as o:
            o.emit(graph)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.pickle'
        return CachedTarget(
            LocalTarget(path, service=NetworkXService)
        )

    def __taskid__(self):
        return 'GraphPruneTask_' +\
                self.name +\
                ('_prune_%d' % self.maxDepth) +\
                ('_types_%s' % '_'.join(
                            [str(v) for v in self.allowedTypes.value]))


if __name__ == '__main__':
    config = {
        "GraphSubTask": {
            "graphPath": "/Users/cedricrichter/Documents/Arbeit/Ranking/PyPRSVT/static/results-tb-raw/",
            "out_dir": "./test2/",
            "cpaChecker": "/Users/cedricrichter/Documents/Arbeit/Ranking/cpachecker"
                },
        "GraphConvertTask": {
            "out_dir": "./test2/"
        },
        "GraphPruningTask": {
            "out_dir": "./test2/"
        },
        "GraphIndexTask": {
            "categories": ['array-examples', 'array-industry-pattern']
        }
            }

    injector = task.ParameterInjector(config)
    planner = task.TaskPlanner(injector=injector)
    exe = task.TaskExecutor()

    name = "3_false-unreach-call_ground.i"
    task = GraphPruningTask(name, 5)
    plan = planner.plan(task)
    exe.executePlan(plan)
    for t in plan.nodes():
        node = plan.node[t]
        if 'task' in node:
            del node['task']
            del node['output']
    data = nx.node_link_data(plan)
    with open('./test/stats.json', 'w') as o:
        json.dump(data, o)
