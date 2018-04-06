from pyTasks import task, remote
from tqdm import tqdm
import networkx as nx


if __name__ == '__main__':
    from gram_tasks import ExtractKernelBagTask
    from prepare_tasks import GraphIndexTask
    from gram_tasks import MDSTask

    config = {
        "GraphSubTask": {
            "graphPath": "/Users/cedricrichter/Documents/Arbeit/Ranking/PyPRSVT/static/results-tb-raw/",
            "graphOut": "./test/",
            "cpaChecker": "/Users/cedricrichter/Documents/Arbeit/Ranking/cpachecker"
                },
        "GraphConvertTask": {
            "graphOut": "./test/"
        },
        "CategoryLookupTask": {
            "graphPaths": "/Users/cedricrichter/Documents/Arbeit/Ranking/PyPRSVT/static/results-tb-raw/"
        },
        "MemcachedTarget": {
            "baseDir": "./cache/"
        },
        "GraphIndexTask": {
            "categories": ["array-examples", "loops", "reducercommutativity"]
        },
        "GraphPruningTask": {
            "graphOut": "./test/"
        }
            }

    injector = task.ParameterInjector(config)
    planner = task.TaskPlanner(injector=injector)
    exe = task.TaskExecutor()

    iTask = GraphIndexTask()
    plan = planner.plan(iTask)
    helper = task.TaskProgressHelper(plan)
    exe.executePlan(plan)

    with helper.output(iTask) as js:
        index = js.query()

    h_Set = [0, 1, 2, 5]

    # plan = nx.read_gpickle('./graph.pickle')
    plan = nx.DiGraph()

    graphs = []

    for k, v in tqdm(index['categories'].items()):

        graphs.extend(v)

    for g in graphs:
        iTask = ExtractKernelBagTask(g, 0, 5)
        plan = planner.plan(iTask, graph=plan)

    nx.write_gpickle(plan, './graph.pickle')

    server = remote.SheduleServer(
        './graph.pickle', config
    )

    registry = remote.buildRegistry()

    cfg = {}

    worker = remote.Worker(
        cfg, registry
    )

    handler = remote.LocalHandler(
        server, [worker]
    )

    handler.run()
