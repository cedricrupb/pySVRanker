from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget, JsonService, ManagedTarget
import svcomp15
import pickle
import base64
import re


class CategoryLookupTask(Task):

    graphPaths = Parameter('./graphs/')
    out_dir = Parameter('./graphs/')
    max_size = Parameter(10000)

    def __init__(self, category):
        self.category = category

    def require(self):
        pass

    def output(self):
        return ManagedTarget(self)

    def __taskid__(self):
        return 'CategoryLookupTask_%s' % self.category

    def run(self):
        results = svcomp15.read_category(
                                            self.graphPaths.value,
                                            self.category,
                                            self.max_size.value
                                        )

        assert results

        with self.output() as o:
            o.emit((self.category, results))


class GraphIndexTask(Task):
    out_dir = Parameter('./index/')
    categories = Parameter(['array-examples'])

    def require(self):
        t = []
        for category in self.categories.value:
            t.append(Optional(CategoryLookupTask(category)))
        return t

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'GraphIndexTask_%s' % containerHash(
                                            self.categories.value
                                        )

    def __extractFiles(self, result):
        files = set([])
        for v in result.values():
            files.update(v.index.values)
        return files

    def __buildDict(self):
        outDict = {}
        filesDict = {}
        for inCat in self.input():
            if inCat is not None:
                with inCat as openCat:
                    category, results = openCat.query()
                    files = self.__extractFiles(results)
                    filesDict[category] = files
                    outDict[category] = results
        return outDict, filesDict

    def _postfix(self, s, l):
        o = s[-l:]
        o = re.sub('[^\w\-_]', '_', o)
        return o

    def __shortestPostfix(self, strlist):
        helper = {}
        minP = 1
        for s in strlist:
            post = minP
            k = self._postfix(s, post)

            while k in helper and post < len(s):
                v = helper[k]
                del helper[k]
                post += 1
                k = self._postfix(v, post)
                helper[k] = v
                k = self._postfix(s, post)

            if post == len(s):
                continue

            helper[k] = s
            minP = post

        return helper

    @staticmethod
    def convert_b64(content):
        catDict = content['stats']
        for k in catDict:
            for i in catDict[k]:
                catDict[k][i] =\
                        pickle.loads(
                            base64.b64decode(catDict[k][i][1:]))

    def run(self):
        catDict, filesDict = self.__buildDict()

        j = []
        for v in filesDict.values():
            j.extend(v)

        indexMap = self.__shortestPostfix(j)
        print(indexMap)
        prefixMap = {v: k for k, v in indexMap.items()}

        for k, v in filesDict.items():
            filesDict[k] = [prefixMap[x] for x in v]

        for k in catDict:
            for i in catDict[k]:
                catDict[k][i] =\
                        str(base64.b64encode(pickle.dumps(catDict[k][i],
                                             pickle.HIGHEST_PROTOCOL)))

        index = {'index': indexMap,
                 'categories': filesDict,
                 'stats': catDict}

        with self.output() as o:
            o.emit(index)
