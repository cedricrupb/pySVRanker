import json
import re
import numpy as np
from scipy.sparse import coo_matrix, diags
from tqdm import tqdm


def detect_category(path):
    reg = re.compile('sv-benchmarks/c/(\D+)/')
    o = reg.search(path)
    if o is None:
        return 'unknown'
    return o.group()[16:-1]


def enumerateable(obj):
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
    counter = 0
    if 'counter' in mapping:
        counter = mapping['counter']

    if key not in mapping:
        mapping[key] = counter
        mapping['counter'] = counter + 1

    return mapping[key]


def read_bag(path):
    with open(path, 'r') as o:
        jsonBag = json.load(o)

    return WLProgramBag(content=jsonBag)


def normalize_gram(GR):

    D = diags(1/np.sqrt(GR.diagonal()))

    return D * GR * D


class WLProgramBag:

    def __init__(self, content={}, init_bags={}, init_categories={}):
        self.bags = init_bags
        self.categories = init_categories
        self._parse_content(content)

    def _parse_content(self, content):
        for k, B in content.items():
            category = 'unknown'
            if 'file' in B:
                category = detect_category(B['file'])
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(k)
            self.bags[k] = B

    def get_categories(self):
        return list(self.categories.keys())

    def get_category(self, category):
        categories = {k: self.categories[k] for k in enumerateable(category)}
        flat = []
        for k, v in categories.items():
            flat.extend(v)
        bags = {k: self.bags[k] for k in flat}
        return WLProgramBag(init_bags=bags, init_categories=categories)

    def _features(self, graphIndex={}, nodeIndex={}):

        row = []
        column = []
        data = []

        K = {}
        for ID, entry in self.bags.items():
            gI = indexMap(ID, graphIndex)
            for n, c in entry['kernel_bag'].items():
                nI = indexMap(n, nodeIndex)
                row.append(gI)
                column.append(nI)
                data.append(c)

        phi = coo_matrix((data, (row, column)),
                         shape=(graphIndex['counter'], nodeIndex['counter']),
                         dtype=np.uint64)

        return graphIndex, nodeIndex, phi.tocsr()

    def features(self, graphIndex={}, nodeIndex={}):
        return self._features(graphIndex, nodeIndex)

    def _dot_gram(self, graphIndex={}):
        graphIndex, _, phi = self._features(graphIndex)
        return graphIndex, phi.dot(phi.transpose())

    @staticmethod
    def pairwise_index(D1, D2):
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
        VX, VY = WLProgramBag.pairwise_index(X, Y)

        return kernel(VX, VY)

    @staticmethod
    def dis_to_sim(X):
        MAX = np.full(X.shape, np.amax(X), dtype=np.float64)

        return MAX - X

    def _custom_gram(self, kernel, graphIndex={}):
        K = {}

        for ID, entry in self.bags.items():
            gI = indexMap(ID, graphIndex)
            K[gI] = entry['kernel_bag']

        T_GR = np.zeros((graphIndex['counter'], graphIndex['counter']),
                        dtype=np.float64)

        for i in tqdm(np.arange(graphIndex['counter'])):
            for j in range(graphIndex['counter']):
                if i <= j:
                    T_GR[i, j] = WLProgramBag.pairwise_kernel(kernel,
                                                              K[i], K[j])
                    T_GR[j, i] = T_GR[i, j]

        if T_GR[0, 0] == 0:
            T_GR = WLProgramBag.dis_to_sim(T_GR)

        return graphIndex, T_GR

    def gram(self, kernel=None, graphIndex={}):
        if kernel is None:
            index, GR = self._dot_gram(graphIndex)
        else:
            index, GR = self._custom_gram(kernel, graphIndex)

        return index, GR

    def normalized_gram(self, kernel=None, graphIndex={}):
        index, GR = self.gram(kernel, graphIndex)

        GR_norm = normalize_gram(GR)

        return index, GR_norm

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

    def indexed_labels(self, graphIndex):
        graphIndex = graphIndex.copy()
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

    def indexed_times(self, graphIndex):
        graphIndex = graphIndex.copy()
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

    def ranking(self, graphIndex):
        return self._prep_y(self.indexed_labels(graphIndex))

    def __add__(self, other):
        if isinstance(other, WLProgramUnion):
            return WLProgramUnion(other, self)
        return WLProgramUnion(self, other)

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

        return WLProgramBag(init_bags=bags,
                            init_categories=self.categories)


class WLProgramUnion(WLProgramBag):

    def __init__(self, bag, union_bag):
        super().__init__(init_bags=union_bag.bags, init_categories=union_bag.categories)
        self._bag = bag
        self._union_bag = union_bag

    def features(self, graphIndex={}):
        out = []
        out_node_index = []
        graphIndex, nodeIndex, features = super()._features(graphIndex)
        out.append(features)
        out_node_index.append(nodeIndex)
        graphIndex, nodeIndices, features = self._bag.features()
        if isinstance(self._bag, WLProgramUnion):
            out.extend(features)
            out_node_index.extend(nodeIndices)
        else:
            out.append(features)
            out_node_index.append(nodeIndices)
        return graphIndex, out_node_index, out

    def gram(self, kernel=None, graphIndex={}):
        graphIndex, gram1 = super().gram(kernel, graphIndex)
        graphIndex, gram2 = self._bag.gram(kernel, graphIndex)

        return graphIndex, gram1 + gram2

    def get_category(self, category):
        return WLProgramUnion(
            self._bag.get_category(category),
            self._union_bag.get_category(category)
        )

    def times(self, indices=None, is_category=False):
        times_bag = self._bag.times(indices, is_category)
        times_union = super().times(indices, is_category)

        out = {}
        for k, t in times_bag.items():
            if k in times_union:
                t = max(t, times_union[k])
            out[k] = t
        return out

    def __add__(self, other):
        if isinstance(other, WLProgramUnion):
            out = WLProgramUnion(self, other._union_bag)
            return out + other._bag
        else:
            return WLProgramUnion(self, other)

    def __len__(self):
        return max(len(self._bag), len(self._union_bag))

    def filter(self, func):
        return WLProgramUnion(
            self._bag.filter(func),
            self._union_bag.filter(func)
        )


if __name__ == '__main__':
    path = '/Users/cedricrichter/Documents/Arbeit/Ranking/bootstrap-scripts/gram/ExtractKernelEntitiesTask_0_5_1994972785.json'

    bag = read_bag(path)

    gI, nI, feat = bag.features()
    print(bag.indexed_labels(gI))
