from pyTasks import task
from pyTasks.task import Task, Parameter, TargetTask, Optional
from pyTasks.task import containerHash
from pyTasks.target import CachedTarget, LocalTarget, NetworkXService, JsonService
from os.path import abspath, join, isdir, isfile
import logging
import subprocess
from subprocess import PIPE
import re
import networkx as nx
from .prepare_tasks import GraphIndexTask
from pyTasks.utils import tick
import os
from .svcomp18 import _extract_expected_status
from .svcomp15 import Status
import shutil
import time


class RunCPATask(Task):
    out_dir = Parameter('./out/graph/')
    cpaChecker = Parameter('./cpa/')
    heap = Parameter("16384m")
    timeout = Parameter(None)
    localize = Parameter(None)

    def __init__(self, name, config, spec):
        self.name = name
        self.config = config
        self.spec = spec

    def require(self):
        return GraphIndexTask()

    def _localize(self, path):
        if self.localize.value is not None:
            return path.replace(self.localize.value[0],
                                self.localize.value[1])
        return path

    def run(self):
        with self.input()[0] as i:
            index = i.query()
            index = index['index']

            if self.name not in index:
                raise ValueError('%s does not exist' % self.name)

            path_to_source = self._localize(abspath(index[self.name]))

        expected_result = _extract_expected_status(path_to_source)
        expected_result = expected_result == Status.true

        __path_to_cpachecker__ = self.cpaChecker.value
        cpash_path = join(__path_to_cpachecker__, 'scripts', 'cpa.sh')
        output = 'output'
        output_path = join(__path_to_cpachecker__, output)
        statistics = join(__path_to_cpachecker__, output, 'Statistics.txt')

        specification = join(__path_to_cpachecker__, self.spec)
        config = join(__path_to_cpachecker__, self.config)

        if not isdir(__path_to_cpachecker__):
            raise ValueError('CPAChecker directory not found: %s' % __path_to_cpachecker__)
        if not (isfile(path_to_source) and (path_to_source.endswith('.i') or path_to_source.endswith('.c'))):
            raise ValueError('path_to_source is no valid filepath. [%s]' % path_to_source)

        if isdir(output_path):
            shutil.rmtree(output_path+"/", ignore_errors=True)

        start_time = time.time()

        try:
            proc = subprocess.run([cpash_path,
                                   '-config', config,
                                   '-spec', specification,
                                   '-heap', self.heap.value,
                                   path_to_source,
                                   '-outputpath', output_path
                                   ],
                                  check=False, stdout=PIPE, stderr=PIPE,
                                  timeout=self.timeout.value)

            S = proc.stdout.decode('utf-8')

        except ValueError as err:
            logging.error(err)
            logging.error(proc.args)
            logging.error(proc.stdout.decode('utf-8'))
            logging.error(proc.stderr.decode('utf-8'))
            raise err

        print(S)
        match_vresult = re.search(r'Verification\sresult:\s([A-Z]+)(\.|,)',  S)
        if match_vresult is None:
            raise ValueError('Invalid output of CPAChecker.')
        analysis_output = match_vresult.group(1)

        if expected_result:
            analysis_output = analysis_output == 'TRUE'
        else:
            analysis_output = analysis_output == 'FALSE'

        analysis_time = time.time() - start_time

        print(analysis_output)
        print(analysis_time)

        with self.output() as o:
            o.emit({
                'spec': self.spec,
                'config': self.config,
                'program': path_to_source,
                'solve': analysis_output,
                'time': analysis_time
            })

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'RunCPATask_' + '_'.join([self.name, self.spec[-10:]])


class BenchSpecTask(Task):
    out_dir = Parameter('./out/graph/')

    def __init__(self, programs, config, spec):
        self.programs = programs
        self.config = config
        self.spec = spec

    def require(self):
        return [RunCPATask(p, self.config, self.spec) for p in self.programs]

    def __taskid__(self):
        return "BenchSpecTask_"+self.spec[-10:]

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def run(self):
        out = {}
        for inp in self.input():
            with inp as result:
                out[result['program']] = result
                del out[result['program']]['program']

        with self.output() as o:
            o.emit(out)
