import inspect
import networkx as nx
from tqdm import tqdm
import json
from .classification_tasks import EvaluationAndSettingTask, CGridTask, hDGridTask
from .classification_tasks import EvaluationTask
import numpy as np

__run__ = True
__graph__ = nx.read_gpickle('./graph.pickle')
index = './index/GraphIndexTask_2003808407.json'
with open(index, 'r') as i:
    index = json.load(i)


def cmd_exit():
    global __run__
    __run__ = False


def buildRegistry():
    registry = {}

    possibles = globals().copy()
    possibles.update(locals())

    for k, v in possibles.items():
        if k.startswith('cmd_'):
            k = k[4:]
            registry[k] = {}
            registry[k]['func'] = v
            registry[k]['args'] = inspect.signature(v).parameters.keys()
            registry[k]['default'] = inspect.signature(v).parameters

    return registry


def get_category(graphs):
    global index
    cat = []
    for k, v in index['categories'].items():
        for g in graphs:
            if g in v:
                cat.append(k)
                break
    return cat


def finish_count(G, n):
    finish = 0
    all_ = 0
    for a in nx.algorithms.dag.ancestors(G, n):
        if G.node[n]['finish']:
            finish += 1
        all_ += 1

    return finish, all_


def get_exception(G, n):
    for a in nx.algorithms.dag.ancestors(G, n):
        node = G.node[a]
        if 'exception' in node:
            return node['exception']


def cmd_status(category=None):
    global __graph__
    G = __graph__

    S = [n for n in G if G.out_degree(n) == 0]
    for n in S:
        cat = get_category(G.node[n]['task'].get_params()['graphs'])

        if category is not None and category not in cat:
            continue

        finish, a = finish_count(G, n)
        label = n[:4] + str(cat)
        pbar = tqdm(total=a, desc=label)
        pbar.update(finish)
        pbar.close()


def cmd_eval(category=None, exception='False'):
    global __graph__
    G = __graph__
    prefix = ''

    S = [n for n in G if G.out_degree(n) == 0]
    for n in S:
        if not isinstance(G.node[n]['task'], EvaluationAndSettingTask):
            continue
        cat = get_category(G.node[n]['task'].get_params()['graphs'])

        if category is not None and category not in cat:
            continue

        node = G.node[n]
        label = n[:4] + str(cat)

        if not node['finish']:
            print('%s not finished' % label)

            if exception == 'True':
                ex = get_exception(G, n)
                if ex is not None:
                    print('Cause: %s' % ex)

            continue

        node['output'].mode = 'r'
        node['output'].path = prefix + node['output'].path

        with node['output'] as o:
            stats = o.query()

        print('Evalution for %s: \n' % label)
        setting = stats['setting']
        for k, v in setting.items():
            print('%s: %s' % (k, str(v)))
        mean = stats['mean']
        for k, v in mean.items():
            print('%s: %4.2f (Std: %4.2f)' % (k, v[0], 2*v[1]))


def add_time(D, name, subname, time):
    if name not in D:
        D[name] = {}
    d = D[name]
    if subname not in d:
        d[subname] = 0.0
    d[subname] += time


def calc_time_c(G, t):
    ignore = set([])

    graph_time = {}

    for a in nx.algorithms.dag.ancestors(G, t):
        node = G.node[a]
        if a.startswith('Graph') and not a.startswith('GraphIndexTask'):
            name = node['task'].get_params()['name']
            if not node['finish'] or name in ignore:
                ignore.add(name)
                continue
            add_time(graph_time, name, 'graph', float(node['time']))
        elif a.startswith('Prepare'):
            name = node['task'].get_params()['graph']
            if not node['finish'] or name in ignore:
                ignore.add(name)
                continue
            add_time(graph_time, name, 'kernel', float(node['time']))
        elif a.startswith('WLKernel') or a.startswith('Normalized'):
            names = node['task'].get_params()['graphs']
            if not node['finish']:
                continue
            div = len(names)
            for name in names:
                add_time(graph_time, name, 'kernel', float(node['time']) / div)
            graph_time['gram'] = float(node['time'])

    return graph_time


def time_c(G, t):
    task = G.node[t]['task']
    params = task.get_params()
    graphs = params['graphs']
    size = len(graphs)
    train_index = params['train_index']
    train_graph = [x for i, x in enumerate(graphs) if i in train_index]

    graph_time = calc_time_c(G, t)
    task_time = float(G.node[t]['time']) / 490
    # print('%s: Time %f' % (t, task_time))
    graph_ratio = []
    test_times = []
    train_time = 0.0

    for k, v in graph_time.items():
        if k == 'gram':
            continue
        time = v['graph'] + v['kernel']
        graph_ratio.append(v['graph'] / time)
        time += task_time
        if k in train_graph:
            train_time += time
        else:
            test_times.append(time+graph_time['gram'])

    return train_time, test_times, graph_ratio, size


def find_gridTask(G, n):
    cGrid = []
    stack = [n]
    while len(stack) > 0:
        act = stack.pop()
        node = G.node[act]['task']

        if isinstance(node, CGridTask):
            cGrid.append(act)
        else:
            for p in G.predecessors(act):
                pNode = G.node[p]['task']
                if isinstance(pNode, EvaluationTask)\
                   or isinstance(pNode, hDGridTask)\
                   or isinstance(pNode, CGridTask):
                    stack.append(p)

    train_times = []
    test_times = []
    graph_ratios = []
    size = 0

    for c in cGrid:
        train, test, graph, s = time_c(G, c)
        size = max(size, s)
        train_times.append(train)
        test_times.extend(test)
        graph_ratios.extend(graph)

    return np.mean(train_times), np.mean(test_times), np.mean(graph_ratios), size


def cmd_time(category=None):
    global __graph__
    G = __graph__
    prefix = '/home/cedricr/common/'

    S = [n for n in G if G.out_degree(n) == 0]
    for n in S:
        if not isinstance(G.node[n]['task'], EvaluationAndSettingTask):
            continue
        cat = get_category(G.node[n]['task'].get_params()['graphs'])

        if category is not None and category not in cat:
            continue

        node = G.node[n]
        if not node['finish']:
            continue

        label = n[:4] + str(cat)

        train, test, graph, size = find_gridTask(G, n)
        print('Time for %s [%d graphs]:' % (label, size))
        print('Average train time: %f' % train)
        print('Average test time: %f' % test)
        print('Average ratio of graph time: %f' % graph)


if __name__ == '__main__':
    d = buildRegistry()

    while __run__:
        s = input('prompt: ')

        cmd_str = s.split(' ')
        if cmd_str[0] in d:
            cmd = d[cmd_str[0]]
            args = cmd_str[1:]
            actual_param = {}
            func_param = cmd['default']

            for i, param in enumerate(cmd['args']):
                if i < len(args):
                    val = args[i]
                    actual_param[param] = val
                elif func_param[param].default is not inspect.Parameter.empty:
                    actual_param[param] = func_param[param].default
                else:
                    print('%s is not given' % param)

            cmd['func'](**actual_param)

        else:
            print('Unknown command %s' % cmd_str[0])
