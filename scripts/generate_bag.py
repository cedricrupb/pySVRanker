import argparse
import networkx as nx
import os
import json
import mmh3
from glob import glob
from tqdm import tqdm
import traceback
import sys


node_relabel = {
    "UNSIGNED_INT": "INT",
    "LONG_UNSIGNED_INT": "LONG",
    "LONG_INT": "LONG",
    "LONGLONG_UNSIGNED_INT": "LONG",
    "LONGLONG_INT": "LONG",
    "LONG_UNSIGNED_LONG": "LONG",
    "LONG_LONG": "LONG",
    "LONGLONG_UNSIGNED_LONGLONG": "LONG",
    "LONGLONG_LONGLONG": "LONG",
    "UNSIGNED_CHAR": "CHAR",
    "VOLATILE_LONG_LONG": "VOLATILE_LONG",
    "VOLATILE_LONG_UNSIGNED_INT": "VOLATILE_LONG",
    "VOLATILE_LONG_INT": "VOLATILE_LONG",
    "VOLATILE_LONG_UNSIGNED_LONG": "VOLATILE_LONG",
    "VOLATILE_UNSIGNED_INT": "VOLATILE_INT",
    "CONST_UNSIGNED_INT": "CONST_INT",
    "CONST_LONG_LONG": "CONST_LONG",
    "CONST_LONG_UNSIGNED_LONG": "CONST_LONG",
    "CONST_LONGLONG_UNSIGNED_LONGLONG": "CONST_LONG",
    "CONST_LONGLONG_LONGLONG": "CONST_LONG",
    "CONST_UNSIGNED_CHAR": "CONST_CHAR",
    "INT_LITERAL_SMALL": "INT_LITERAL",
    "INT_LITERAL_MEDIUM": "INT_LITERAL",
    "INT_LITERAL_LARGE": "INT_LITERAL"
}

edge_relabel = {
    'cd_t': 'cd',
    'cd_f': 'cd',
    'dummy': 'cfg'
}


def __relabel(G, n, relabel={}):
    global node_relabel
    global edge_relabel

    source_label = G.nodes[n]['label']

    if n in relabel:
        source_label = relabel[n]
    if source_label in node_relabel:
        source_label = node_relabel[source_label]

    neighbours = []

    for u, _, d in G.in_edges(n, keys=True):
        source = G.nodes[u]['label']

        if u in relabel:
            source = relabel[u]
        if source in node_relabel:
            source = node_relabel[source]

        edge_t = d

        if edge_t in edge_relabel:
            edge_t = edge_relabel[edge_t]

        neighbours.append(
            str(mmh3.hash(
                '_'.join(
                    [str(t) for t in [source, edge_t]]
                )
            ))
        )

    if len(neighbours) > 0:
        neighbours = sorted(neighbours)
        neighbours = '_'.join(neighbours)
        source = source_label + "_" + neighbours
    else:
        source = source_label

    return str(mmh3.hash(
        source
    ))


def wlGraphRelabel(G, relabel={}):
    next_relabel = {}

    for n in G:
        next_relabel[n] = __relabel(G, n, relabel)

    return next_relabel


def labelCount(G, relabel):
    count = {}

    for n in G:
        if n in relabel:
            label = relabel[n]
        else:
            label = G.nodes[n]['label']
        if label not in count:
            count[label] = 0
        count[label] += 1

    return count


def labelDepth(G):
    astNodes = set([])

    for u, _, d in G.out_edges(keys=True):
        if d == "s" and 'depth' not in G.nodes[u]:
            astNodes.add(u)

    if len(astNodes) == 0:
        return

    remove = set([])
    for u, v, d in G.out_edges(astNodes, keys=True):
        if v not in astNodes:
            G.nodes[u]['depth'] = 1
            remove.add(u)

    for r in remove:
        astNodes.remove(r)

    while len(astNodes) > 0:
        remove = set([])
        for u, v in G.out_edges(astNodes):
            attr = G.nodes[v]
            if 'depth' in attr:
                G.nodes[u]['depth'] = attr['depth'] + 1
                remove.add(u)

        for r in remove:
            astNodes.remove(r)

    for n in G:
        attr = G.nodes[n]
        if 'depth' not in attr:
            attr['depth'] = 0


def truncate(G, k):
    labelDepth(G)

    remove = set([])
    for n in G:
        if G.nodes[n]['depth'] > k:
            remove.add(n)

    if len(remove) > 0:
        G.remove_nodes_from(remove)


def is_forward_and_parse(e):
    if e.endswith('|>'):
        return e[:-2], True
    return e[2:], False


def parse_dfs_nx(R):
    if R is None:
        return nx.MultiDiGraph()
    graph = nx.MultiDiGraph()

    for _R in R:
        graph.add_node(_R[0], label=_R[2])
        graph.add_node(_R[1], label=_R[4])
        e_label, forward = is_forward_and_parse(_R[3])
        if forward:
            graph.add_edge(_R[0], _R[1], key=e_label)
        else:
            graph.add_edge(_R[1], _R[0], key=e_label)

    return graph


def pathId(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def load_graph(path):
    if not os.path.isfile(path):
        raise ValueError("Unknown path: %s" % path)
    with open(path, "r") as i:
        R = json.load(i)
    G = parse_dfs_nx(R)
    G.graph['id'] = pathId(path)
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to transform graphs in bag representation.")
    parser.add_argument("input", help="input file or directory", type=str)
    parser.add_argument("iteration", help="iteration bound for WL labelling",
                        type=int)
    parser.add_argument("depth", help="depth bound for AST tree", type=int)
    parser.add_argument("output", help="output directory", type=str)
    parser.add_argument("-b", "--bunch", action="store_true")

    args = parser.parse_args()

    if args.iteration < 0:
        print("Iteration bound has to be positive. Exit")
        exit()

    if args.depth < 0:
        print("AST depth has to be positive. Exit")
        exit()

    if not os.path.exists(args.input):
        print("%s doesn't exist. Exit")
        exit()

    inputs = [args.input]
    if args.bunch:
        inputs = glob(os.path.join(args.input, "*.dfs"))

    for inp in inputs:
        with tqdm(total=args.iteration+3) as pbar:
            try:
                G = load_graph(inp)
            except Exception as e:
                print("Problem while loading: %s. Continue with next." % (inp))
                traceback.print_exc(file=sys.stdout)

            pbar.update(1)

            truncate(G, args.depth)

            pbar.update(1)

            relabel = node_relabel
            for i in range(args.iteration + 1):
                C = labelCount(G, relabel)

                with open(os.path.join(args.output, '%s_%d_%d.json' % (G.graph['id'], i, args.depth)), 'w') as o:
                    json.dump({
                        'file': inp,
                        'kernel_bag': C
                    }, o, indent=4)

                pbar.update(1)

                if i < args.iteration:
                    relabel = wlGraphRelabel(G, relabel)
