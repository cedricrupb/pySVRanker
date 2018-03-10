from pyTasks import task
from pyTasks.task import Task, Parameter, TaskProgressHelper
from pyTasks.task import containerHash
from pyTasks.target import CachedTarget, LocalTarget, JsonService
from prepare_tasks import GraphIndexTask
from svcomp15 import Status
import pandas as pd
import math


class DefineClassTask(Task):
    out_dir = Parameter('./ranking/')

    def __init__(self, graphs):
        self.graphs = graphs

    def require(self):
        return GraphIndexTask()

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.csv'
        return CachedTarget(
                            LocalTarget(path, service=JsonService)
                            )

    def __taskid__(self):
        return "DefineClassTask_%s" % containerHash(self.graphs)

    def run(self):
        with self.input()[0] as i:
            index = i.query()
            GraphIndexTask.convert_b64(index)

        tools = index['stats']

        ranking = {}
        for g in self.graphs:
            gKey = index['index'][g]
            time_rank = []
            score_rank = {}
            max_score = -20
            for cat, D in tools.items():
                for t, D in D.items():
                    if gKey not in D.index:
                        continue
                    d = D.loc[gKey]
                    status = d['status']
                    expected = d['expected_status']
                    state = 'false'
                    if (status is Status.false and expected is Status.false)\
                       or (status is Status.true and expected is Status.true):
                        state = 'correct'
                    score_rank[t] = state

                    time = d['cputime']
                    time_rank.append((t, time))

            time_tupel = []
            for i, (t1, time1) in enumerate(time_rank):
                for j, (t2, time2) in enumerate(time_rank):
                    if i < j:
                        tupel = None
                        if math.fabs(time1 - time2) <= time1 * 0.01:
                            tupel = (t1, False, t2)
                        elif time1 >= 900 and time2 >= 900:
                            tupel = (t1, False, t2)
                        elif time1 < time2:
                            tupel = (t1, True, t2)
                        else:
                            tupel = (t2, True, t1)
                        time_tupel.append(tupel)

            ranking[g] = {'score': score_rank, 'time_rank': time_tupel}

        with self.output() as pdfile:
            pdfile.emit(ranking)


if __name__ == '__main__':
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
            "categories": ['array-examples', 'array-industry-pattern']
        }
            }

    injector = task.ParameterInjector(config)
    planner = task.TaskPlanner(injector=injector)
    exe = task.TaskExecutor()

    task = GraphIndexTask()
    plan = planner.plan(task)
    tph = TaskProgressHelper(plan)
    exe.executePlan(plan)

    with tph.output(task) as js:
        index = js.query()

    graphs = index['categories']['array-examples'].copy()
    graphs.extend(index['categories']['array-industry-pattern'])
    task = DefineClassTask(graphs)
    plan = planner.plan(task, graph=plan)
    exe.executePlan(plan)
