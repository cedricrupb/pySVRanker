from pyTasks.task import Parameter, Task
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, BulkWriteError
from pyTasks.target import LocalTarget, JsonService
from pyTasks.utils import containerHash
from .graph_tasks import GraphPruningTask
from .bag import indexMap, normalize_gram, detect_category
from scipy.sparse import coo_matrix, diags
from .kernel_function import jaccard_kernel
from .prepare_tasks import GraphIndexTask
from .ranking_task import ExtractInfoTask
import os
import numpy as np
import gridfs
import networkx as nx
import tempfile
try:
    import mmh3
except ImportError:
    import pymmh3 as mmh3


__client__ = {}


def setup_client(url, auth=None):
    global __client__
    if url not in __client__:
        if auth is not None:
            uri = 'mongodb://%s:%s@%s/%s' % (
                quote_plus(auth['username']),
                quote_plus(auth['password']),
                url,
                auth['authSource']
            )
        else:
            uri = 'mongodb://%s/' % url

        __client__[url] = MongoClient(uri)
    return __client__[url]


def path_split(path):
    category = detect_category(path)
    f = os.path.basename(path)
    f = os.path.splitext(f)[0]
    return category, f


class MongoResource:

    def __init__(self, collection, index, key):
        self.collection = collection
        self.index = index
        self.key = key

    def emit(self, obj):
        obj[self.index] = self.key
        return self.collection.insert_one(obj)

    def query(self):
        return self.collection.find_one({self.index: self.key})


class MongoResourceTarget:
    auth = Parameter(None)
    url = Parameter('example.org')
    database = Parameter('db')

    def __init__(self, collection_name, index, key):
        self.collection_name = collection_name
        self.index = index
        self.key = key

    def __enter__(self):
        self._connection = setup_client(self.url.value, self.auth.value)
        self.db = self._connection[self.database.value]
        self.collection = self.db[self.collection_name]
        resource = MongoResource(self.collection, self.index, self.key)
        resource.db = self.db
        return resource

    def __exit__(self, type, value, tb):
        self.collection = None

    def exists(self):
        return False

    def __getstate__(self):
        return {
            'auth': self.auth.value,
            'url': self.url.value,
            'db': self.database.value,
            'collection': self.collection_name,
            'index': self.index,
            'key': self.key
        }

    def __setstate__(self, state):
        self.auth.value = state['auth']
        self.url.value = state['url']
        self.database.value = state['db']
        self.collection_name = state['collection']
        self.index = state['index']
        self.key = state['key']


class MongoGraphTask(Task):
    collection = Parameter('graphs')

    def __init__(self, name, maxDepth):
        self.name = name
        self.maxDepth = maxDepth

    def require(self):
        return [GraphIndexTask(),
                GraphPruningTask(self.name, self.maxDepth)]

    def __taskid__(self):
        return 'MongoGraphTask_%s_%d' % (
            self.name, self.maxDepth
        )

    def output(self):
        return MongoResourceTarget(
            self.collection.value,
            '_id', self.name
        )

    def run(self):
        with self.input()[0] as inp:
            index = inp.query()['index']

        with self.input()[1] as inp:
            G = inp.query()

        category, f_name = path_split(index[self.name])

        out = {
            'category': category,
            'file': f_name,
            'ast_depth': self.maxDepth,
            'node_count': G.number_of_nodes(),
            'edge_count': G.size()
        }

        degree = np.array(
            [x[1] for x in G.in_degree()]
        )

        out.update({
            'min_in_degree': float(degree.min()),
            'max_in_degree': float(degree.max()),
            'mean_in_degree': float(degree.mean())
        })

        degree = np.array(
            [x[1] for x in G.out_degree()]
        )

        out.update({
            'min_out_degree': float(degree.min()),
            'max_out_degree': float(degree.max()),
            'mean_out_degree': float(degree.mean())
        })

        with tempfile.NamedTemporaryFile() as fp:
            nx.readwrite.write_gexf(
                clean_edges(G), fp.name
            )
            fp.seek(0)
            with self.output() as o:
                fs = gridfs.GridFS(o.db)
                out['ref'] = fs.put(fp)
                o.emit(out)


class MongoGraphLabelTask(Task):
    collection = Parameter('graph_label')

    def __init__(self, graphs):
        self.graphs = graphs

    def require(self):
        return [ExtractInfoTask(self.graphs)]

    def __taskid__(self):
        return 'MongoGraphLabelTask'

    def output(self):
        return MongoResourceTarget(
            self.collection.value,
            '_id', 'labels'
        )

    def run(self):
        with self.input()[0] as inp:
            info = inp.query()

        bulk = []
        for graph, V in info.items():
            for tool, result in V.items():
                bulk.append(
                    {
                        'graph_id': graph,
                        'tool': tool,
                        'label': result
                    }
                )

        with self.output() as mongo:
            try:
                mongo.collection.insert_many(bulk)
            except BulkWriteError:
                pass


def _build_labels(G, h, mapping, id2node):
    act_map = mapping[h - 1]
    next_map = {}
    next_id_map = {}
    act_id_map = {}
    if h > 1:
        act_id_map = id2node[h - 2]

    labels = {}

    for u, v, d in G.in_edges(data=True):

        if v not in next_id_map:
            next_id_map[v] = set([])
        neigh = [u]
        if u in act_id_map:
            neigh = act_id_map[u]
        next_id_map[v].update(neigh)

        if v not in labels:
            labels[v] = []

        source = act_map[u]
        edge_t = d['type']
        truth = d['truth']

        long_edge_label = str(mmh3.hash(
                '_'.join(
                    [str(t) for t in [source, edge_t, truth]
                     ]
                )
            ))

        labels[v].append(long_edge_label)

    for v, N in labels.items():
        next_map[v] = G.node[v]['label'] + '_' + str(
            mmh3.hash(
                act_map[v] + '_' + '_'.join(
                    sorted(N)
                )
            )
        )

    next_map.update({
        k: act_map[k] for k in act_map if k not in next_map
    })

    mapping[h] = next_map
    id2node[h - 1] = next_id_map


def clean_edges(G):
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k]['type'] = G.edges[u, v, k]['type'].name
        G.edges[u, v, k]['truth'] = G.edges[u, v, k]['truth'].name
    return G


class MongoWLTask(Task):
    collection = Parameter('wl_graph')
    id_collection = Parameter('id_sub')
    map_collection = Parameter('map_graph')

    def __init__(self, name, h, maxDepth):
        self.name = name
        self.h = h
        self.maxDepth = maxDepth

    def require(self):
        return GraphPruningTask(self.name, self.maxDepth)

    def __taskid__(self):
        return 'MongoWLTask_%s_%d_%d' % (
            self.name, self.h, self.maxDepth
        )

    def output(self):
        return MongoResourceTarget(
            self.collection.value,
            'graph_id', self.name
        )

    def run(self):
        with self.input()[0] as inp:
            G = inp.query()

        mapping = [None]*(self.h + 1)
        id2node = [None]*self.h
        mapping[0] = {n: G.node[n]['label'] for n in G}

        for i in range(1, self.h+1):
            _build_labels(G, i, mapping, id2node)

        attr = {
            'source': 'source',
            'target': 'target',
            'name': 'id',
            'key': 'key',
            'link': 'edges'
        }

        with self.output() as mongo:
            id_coll = mongo.db[self.id_collection.value]

            for i, idnode in enumerate(id2node):
                act_map = mapping[i + 1]
                seen = set([])
                for node, neigh in idnode.items():
                    label = act_map[node]
                    if label in seen:
                        continue
                    subgraph = nx.readwrite.json_graph.node_link_data(
                                clean_edges(G.subgraph(neigh).copy()), attr)
                    try:
                        id_coll.insert_one(
                            {
                                '_id': label,
                                'h': i + 1,
                                'subgraph': subgraph
                            }
                        )
                    except DuplicateKeyError:
                        pass
                    seen.add(label)

            del id2node

            map_coll = mongo.db[self.map_collection.value]

            try:
                map_coll.insert_many([
                    {
                        '_id': self.name + '_' + str(i),
                        'graph_id': self.name,
                        'h': i,
                        'mapping': map
                    }
                    for i, map in enumerate(mapping)
                ])
            except BulkWriteError:
                pass

            for i, mapped in enumerate(mapping):
                count = {}
                for n in G:
                    label = mapped[n]
                    if label not in count:
                        count[label] = 0
                    count[label] += 1
                try:
                    mongo.emit({
                        '_id': self.name + '_' + str(i),
                        'h': i,
                        'count': count
                        }
                    )
                except DuplicateKeyError:
                    pass


class CleanMongoTask(Task):

    def __init__(self, collection):
        self.collection = collection

    def require(self):
        return []

    def __taskid__(self):
        return 'CleanMongo'

    def output(self):
        return MongoResourceTarget(
            self.collection,
            '_id', 'clean'
        )

    def run(self):
        pass


class MongoSimTask(Task):
    collection = Parameter('graph_sim')
    used_kernel = Parameter('jaccard')

    def __init__(self, graphs, h, maxDepth):
        self.graphs = graphs
        self.h = h
        self.maxDepth = maxDepth

    def require(self):
        return [
            MongoWLTask(g, self.h, self.maxDepth)
            for g in self.graphs
        ]

    def __taskid__(self):
        return 'MongoSimTask_%d_%d' % (
            self.h, self.maxDepth
        )

    def output(self):
        return MongoResourceTarget(
            self.collection.value,
            'graph_id', 'graphs'
        )

    def run(self):
        with self.input()[0] as mongo_inp:
                coll = mongo_inp.collection
                for i in range(self.h + 1):
                    graphIndex = {}
                    nodeIndex = {}

                    row = []
                    column = []
                    data = []

                    for g in self.graphs:
                        s = '%s_%d' % (g, i)
                        wl_graph = coll.find_one({'_id': s})
                        if wl_graph is None:
                            continue
                        gI = indexMap(g, graphIndex)
                        count = wl_graph['count']
                        for n, c in count.items():
                            nI = indexMap(n, nodeIndex)
                            row.append(gI)
                            column.append(nI)
                            data.append(c)

                    phi = coo_matrix((data, (row, column)),
                                     shape=(graphIndex['counter'],
                                            nodeIndex['counter'])).tocsr()

                    phi = normalize_gram(jaccard_kernel(phi))

                    del graphIndex['counter']

                    inv_graphIndex = np.array(
                        [x[0] for x in sorted(list(graphIndex.items()), key=lambda x: x[1])]
                    )

                    used_kernel = self.used_kernel.value

                    bulk = []

                    for gI in range(inv_graphIndex.shape[0]):
                        g = inv_graphIndex[gI]
                        for gJ in range(inv_graphIndex.shape[0]):
                            if gI < gJ:
                                p = inv_graphIndex[gJ]
                                bulk.append({
                                    '_id': containerHash([g, p, i, used_kernel]),
                                    'first_id': g,
                                    'second_id': p,
                                    'h': i,
                                    'sim_function': used_kernel,
                                    'similarity': phi[gI, gJ]
                                })

                    with self.output() as mongo_out:
                        mongo_out.collection.insert_many(bulk)
