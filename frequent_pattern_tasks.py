from pyTasks.tasks import Task, Parameter
from pyTasks.utils import containerHash
from .graph_tasks import GraphPruningTask
from .mongo_tasks import MongoResourceTarget
from sklearn.model_selection import KFold
import numpy as np
from bson.code import Code


def non_filter(label):
    return False


def identity(obj):
    return obj


class MongoGraphNodesTask(Task):
    collection = Parameter("graph_nodes")

    def __init__(self, graph, D):
        self.graph = graph
        self.D = D

    def require(self):
        return GraphPruningTask(self.graph, self.D)

    def __taskid__(self):
        return "GraphNodesTask_%s_%d_%d" % (self.graph, self.D)

    def output(self):
        return MongoResourceTarget(
            self.collection.value, '_id', self.graph
        )

    def run(self):
        with self.input()[0] as i:
            G = i.query()

        nodes = set([])
        for node in G:
            label = G.node[node]['label']
            nodes.add(label)

        with self.output() as o:
            coll = o.collection
            coll.insert_many([
                {'graph_id': self.graph,
                 'node': n}
                for n in nodes
            ])


class MongoFrequencyTask(Task):
    collection = Parameter("node_frequency")

    def __init__(self, graphs, it, D):
        self.graphs = graphs
        self.it = it
        self.D = D

    def require(self):
        return [
            MongoGraphNodesTask(g, self.D)
            for g in self.graphs
        ]

    def output(self):
        return MongoResourceTarget(
            self.collection.value, '_id', 'frequency_%d' % self.it
        )

    def run(self):
        with self.input()[0] as i:
            coll = i.collection

            map = Code("""
                    function(){
                        emit(this.node, 1);
                    }
            """)

            reduce = Code("""
                function(key, values){
                    var total = 0;
                    for(var i = 0; i < values.length; i++){
                        total += values[i];
                    }
                    return total;
                }
            """)

            reduce = coll.map_reduce(map, reduce, self.collection.value)

            all = len(self.graphs)

            reduce.update({}, {'$mul': {'value': 1/all}})
