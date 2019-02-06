"""Script to run experiments locally."""
from pyTasks import task
import argparse
import json
import networkx as nx
import os
from pySVRanker.bag_tasks import BagFilterTask
from tqdm import tqdm
from pySVRanker.czech_tasks import CVCzechSingleEvalutionTask
from pySVRanker.majority_tasks import CVMajorityEvalutionTask


# Execution config

# Number of tools used in the set of labels
number_of_tools = 10

# Depth of AST tree (e.g. 1 ==> AST tree is cut after depth 1)
depths = [1, 2, 5]

# Iteration bound for WL kernel (e.g 1 ==> Look at subtrees of height 0 and 1)
iteration = [0, 1, 2]

# Slack variable for Support Vector Machine
C = [0.01, 0.1, 1, 10, 100, 1000]

# Type of programs to look at
types = ['reach', 'termination', 'memory', 'overflow']

# Used measures for evaluation
measures = ['spearmann', 'spearmann_lambda', 'kendall_tau']


def size_of_bags(plan, planner, injector, prefix=None):
    global types
    global iteration
    global depths

    sizes = {}
    ex = task.TaskExecutor(planner, prefix=prefix)

    tasks = {}
    for t in types:
        tasks[t] = {}
        for d in depths:
            indexTask = BagFilterTask(
                0, d, task_type=t, by_id=True
            )
            tasks[t][d] = indexTask
            injector.inject(indexTask)

            planner.plan(indexTask, graph=plan)

            for i in range(1, max(iteration)):
                indexTask = BagFilterTask(
                    i, d, task_type=t, by_id=True
                )
                injector.inject(indexTask)

                planner.plan(indexTask, graph=plan)

    ex.executePlan(plan)

    for t in types:
        for d in depths:
            out = tasks[t][d].output()
            path = out.path
            if prefix:
                path = os.path.join(prefix, path)

            with open(path, 'r') as js:
                index = json.load(js)

            if t not in sizes:
                sizes[t] = {}

            sizes[t][d] = len(list(index.keys()))

    return sizes


def make_tasks(h, D, entities, task_type):
    global C
    global measures
    global number_of_tools
    return CVCzechSingleEvalutionTask(
        number_of_tools, C, h, D, 'f1',
        measures,
        'spearmann', entities,
        task_type=task_type,
        kernel='jaccard', overall=True
    ), CVMajorityEvalutionTask(
        measures, entities, task_type=task_type
    )


def easy_tasks(i, d, type, bag_size):
    return make_tasks(
        i, d, bag_size[type][d], task_type=type
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-p', '--prefix', type=str, required=True)

    args = parser.parse_args()

    if os.path.isfile(args.out):
        print("Load Plan from %s" % args.out)
        plan = nx.read_gpickle(args.out)
    else:
        print("Plan execution.")
        plan = nx.DiGraph()

        with open(args.config, 'r') as cfg_file:
            config = json.load(cfg_file)

        injector = task.ParameterInjector(config)
        planner = task.TaskPlanner(injector)

        bag_size = size_of_bags(plan, planner, injector, args.prefix)
        print(bag_size)

        for d in depths:
            for i in tqdm(iteration):
                for t in types:
                    svm, majority = make_tasks(
                        i, d, bag_size[t][d], task_type=t
                    )
                    planner.plan(svm, graph=plan)
                    planner.plan(majority, graph=plan)

    try:
        ex = task.TaskExecutor(planner, prefix=args.prefix)
        ex.executePlan(plan)
    except KeyboardInterrupt:
        print('Save graph to %s' % args.out)
        nx.write_gpickle(plan, args.out)
