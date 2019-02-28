"""General package to handle programs as bags representation."""
import json
import re
import numpy as np
from scipy.sparse import coo_matrix, diags
from tqdm import tqdm
from .kernel_function import is_pairwise, is_absolute
from .prepare_tasks import select_svcomp
from .svcomp15 import MissingPropertyTypeException


def detect_task_type(svcomp, path):
    """Detect a task type found in program path."""
    svcomp = select_svcomp(svcomp)
    try:
        return svcomp.set_of_properties(path)
    except MissingPropertyTypeException:
        print('Problem with property. Ignore')
        return None


def detect_category(path):
    """Detect a category found in program path."""
    reg = re.compile('sv-benchmarks\/c\/[^\/]+\/')
    o = reg.search(path)
    if o is None:
        return 'unknown'
    return o.group()[16:-1]


def enumerateable(obj):
    """
    Check if object is an enumerateable.

    If yes: return the object.
    If no: Create an enumerateable containing only the object
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    try:
        _ = (e for e in obj)
        return obj
    except TypeError:
        return [obj]


def indexMap(key, mapping):
    """
    Return an existing or newly generated index.

    Caution: A fake entry "counter" is generated for storing the last id.
    This could lead to problems if this is not handled later on.
    """
    counter = 0
    if 'counter' in mapping:
        counter = mapping['counter']

    if key not in mapping:
        mapping[key] = counter
        mapping['counter'] = counter + 1

    return mapping[key]


def read_bag(path):
    """Read a bag representation from Json."""
    with open(path, 'r') as o:
        jsonBag = json.load(o)

    return ProgramBags(content=jsonBag)


def normalize_gram(GR):
    """Normalize a given gram matrix"""

    D = diags(1/np.sqrt(GR.diagonal()))

    return D * GR * D


class ProgramBags:
    """A general class to store and handle programs as bags."""

    def __init__(self, content={}, init_bags={}, init_categories={},
                 svcomp='svcomp18'):
        """
        Init bags.

        content: The bag representation as dictionary.
                 May be parsed from a json file.
        init_bags: Some initial prepared bags.
        init_categories: Some initial prepared categories.
        svcomp: The SV-Comp for parsing categories and task_types
        """
        self.bags = init_bags
        self.categories = init_categories
        self.graphIndex = {}
        self.nodeIndex = {}
        self.svcomp = svcomp
        self._parse_content(content)
        self._index_bags()

    def _parse_content(self, content):
        for k, B in content.items():
            category = 'unknown'
            if 'file' in B:
                category = detect_category(B['file'])
                B['task_type'] = detect_task_type(self.svcomp, B['file'])
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(k)
            self.bags[k] = B

    def _index_bags(self):
        for k in self.bags:
            indexMap(k, self.graphIndex)

    def get_categories(self):
        """Get a list of all categories."""
        return list(self.categories.keys())

    def get_category(self, category):
        """
        Create a new bag only containing tasks of a given category.

        category: One category or a list of categories.
        """
        categories = {k: self.categories[k] for k in enumerateable(category)}
        flat = []
        for k, v in categories.items():
            flat.extend(v)
        bags = {k: self.bags[k] for k in flat}
        return ProgramBags(init_bags=bags, init_categories=categories,
                           svcomp=self.svcomp)

    def get_task_type(self, task_type):
        """Create a new bag only containing tasks of a given task_type."""
        categories = {}
        bags = {}
        for k, V in self.categories.items():
            categories[k] = []
            for v in V:
                task = self.bags[v]
                if task_type in task['task_type']:
                    categories[k].append(v)
                    bags[v] = task
        if len(bags) == 0:
            raise ValueError('No representation found for %s. Fail.'
                             % str(task_type))
        return ProgramBags(init_bags=bags, init_categories=categories,
                           svcomp=self.svcomp)

    def features(self):
        """
        Return a sparse array representing a feature matrix.

        (r, c) is the frequency of a label c in a task r.
        Return format: Sparse CSR Matrix
        """
        row = []
        column = []
        data = []
        floatType = False

        for ID, entry in self.bags.items():
            gI = indexMap(ID, self.graphIndex)
            for n, c in entry['kernel_bag'].items():
                nI = indexMap(n, self.nodeIndex)
                floatType = floatType or isinstance(c, float)
                row.append(gI)
                column.append(nI)
                data.append(c)

        dtype = np.float64 if floatType else np.uint64

        phi = coo_matrix((data, (row, column)),
                         shape=(self.graphIndex['counter'],
                                self.nodeIndex['counter']),
                         dtype=dtype)

        return phi.tocsr()

    def _dot_gram(self):
        phi = self.features()
        return phi.dot(phi.transpose())

    @staticmethod
    def pairwise_index(D1, D2):
        """Index features only between two tasks."""
        index = {}
        O1 = {}

        for d, v in D1.items():
            O1[indexMap(d, index)] = v

        O2 = {}

        for d, v in D2.items():
            O2[indexMap(d, index)] = v

        V1 = np.zeros((index['counter']), dtype=np.int64)

        for o, v in O1.items():
            V1[o] = v

        V2 = np.zeros((index['counter']), dtype=np.int64)

        for o, v in O2.items():
            V2[o] = v

        return V1, V2

    @staticmethod
    def pairwise_kernel(kernel, X, Y):
        """Generate a kernel only between two tasks."""
        VX, VY = ProgramBags.pairwise_index(X, Y)

        return kernel(VX, VY)

    @staticmethod
    def dis_to_sim(X):
        """Return a similarity measure by using a distance measure."""
        MAX = np.full(X.shape, np.amax(X), dtype=np.float64)

        return MAX - X

    def _pairwise_gram(self, kernel):
        K = {}

        for ID, entry in self.bags.items():
            gI = indexMap(ID, self.graphIndex)
            K[gI] = entry['kernel_bag']

        T_GR = np.zeros((self.graphIndex['counter'],
                         self.graphIndex['counter']),
                        dtype=np.float64)

        E = sorted(list(K.keys()))

        for i in tqdm(E):
            for j in E:
                if i <= j:
                    T_GR[i, j] = ProgramBags.pairwise_kernel(kernel,
                                                             K[i], K[j])

                    T_GR[j, i] = T_GR[i, j]

        if T_GR[0, 0] == 0:
            T_GR = ProgramBags.dis_to_sim(T_GR)

        return T_GR

    def _custom_gram(self, kernel):
        if is_pairwise(kernel):
            return self._pairwise_gram(kernel)
        elif is_absolute(kernel):
            return kernel(self.features())
        else:
            raise ValueError('Kernel has to accept 1 (complete feature set)' +
                             ' or 2 (pairwise 1-D) matrices')

    def gram(self, kernel=None):
        if kernel is None:
            GR = self._dot_gram()
        else:
            GR = self._custom_gram(kernel)

        return GR

    def normalized_gram(self, kernel=None):
        return normalize_gram(self.gram(kernel))

    def labels(self, indices=None, is_category=False):
        if indices is None:
            indices = list(self.bags.keys())

        if is_category:
            tmp = indices
            indices = []
            for c in tmp:
                indices.extend(self.categories[c])

        out = {}
        for index in enumerateable(indices):
            out[index] = self.bags[index]['label']

        return out

    def indexed_labels(self):
        graphIndex = self.graphIndex.copy()
        counter = graphIndex['counter']
        del graphIndex['counter']

        indices = list(graphIndex.keys())
        labels = self.labels(indices=indices)

        y = [None] * counter

        for gI, index in graphIndex.items():
            y[index] = labels[gI]

        return y

    def times(self, indices=None, is_category=False):
        if indices is None:
            indices = list(self.bags.keys())

        if is_category:
            tmp = indices
            indices = []
            for c in tmp:
                indices.extend(self.categories[c])

        out = {}
        for index in enumerateable(indices):
            if 'time' in self.bags[index]:
                out[index] = self.bags[index]['time']

        return out

    def indexed_times(self):
        graphIndex = self.graphIndex.copy()
        counter = graphIndex['counter']
        del graphIndex['counter']

        indices = list(graphIndex.keys())
        times = self.times(indices=indices)

        y = [-1.0] * counter

        for gI, index in graphIndex.items():
            if gI in times:
                y[index] = times[gI]

        return y

    def _incr(self, d, k, i=1):
        if k not in d:
            d[k] = 0
        d[k] += i

    def _prep_y(self, y):
        out_y = []
        for _y in y:
            count = {}
            for i, t in enumerate(_y):
                for j, o in enumerate(_y):
                    if i < j:
                        time_t = _y[t]['time']
                        time_o = _y[o]['time']

                        tbetter = time_t < time_o

                        solve_t = _y[t]['solve'] == 'true'
                        solve_o = _y[o]['solve'] == 'true'
                        if solve_t and not solve_o:
                            self._incr(count, t)
                            self._incr(count, o, 0)
                        elif solve_o and not solve_t:
                            self._incr(count, o)
                            self._incr(count, t, 0)
                        elif tbetter:
                            self._incr(count, t)
                            self._incr(count, o, 0)
                        else:
                            self._incr(count, o)
                            self._incr(count, t, 0)

            d = [x[0] for x in
                 sorted(list(count.items()),
                        key=lambda y: y[1],
                        reverse=True)
                 ]
            out_y.append(d)

        return out_y

    def ranking(self):
        return self._prep_y(self.indexed_labels())

    def __len__(self):
        return len(self.bags)

    def filter(self, func):
        bags = {}
        for k, Bag in self.bags.items():
            D = {'label': Bag['label']}
            if 'time' in Bag:
                D['time'] = Bag['time']
            if func(D):
                bags[k] = Bag

        return ProgramBags(init_bags=bags,
                           init_categories=self.categories)
