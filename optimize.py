import networkx as nx
from .graph_tasks import GraphPruningTask, GraphConvertTask


def optimizePruning(graph):
    pruneGraph = {}

    for n in graph:
        nTask = graph.node[n]['task']
        if isinstance(nTask, GraphPruningTask):
            for p in graph.predecessors(n):
                pTask = graph.node[p]['task']
                if isinstance(pTask, GraphConvertTask):
                    pred = p
                    break
            name = nTask.name
            depth = nTask.maxDepth

            if name not in pruneGraph:
                pruneGraph[name] = []

            pruneGraph[name].append((n, pred, depth))

    for n, D in pruneGraph.items():
        D = sorted(D, key=lambda x: x[2], reverse=True)
        for i in range(1, len(D)):
            prune, pred, _ = D[i]
            nPred, _, _ = D[i-1]
            graph.remove_edge(pred, prune)
            graph.add_edge(nPred, prune)
