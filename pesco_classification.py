from pyTasks.task import Task, Parameter, Optional
from pyTasks.target import LocalTarget, JsonService, FileTarget
import numpy as np
from .bag_tasks import index
from .rank_scores import select_score

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC


def create_index(slice_index, graphIndex, labelIndex, properties):

    X_index = []
    y_index = []

    for k, i in ((k, i) for k, i in graphIndex.items() if i in slice_index):
        for prop in properties:
            if k in labelIndex[prop]:
                X_index.append(i)
                y_index.append(
                    labelIndex[prop][k]
                )

    if len(properties) == 1 and len(X_index) < len(slice_index):
        raise ValueError("Too few examples: %d != %d" % (len(X_index), len(slice_index)))

    return X_index, y_index


class LoadJsonTask(Task):

    def __init__(self, path):
        self.path = path

    def require(self):
        pass

    def output(self):
        return LocalTarget(
            self.path, service=JsonService
        )

    def run(self):
        pass


class RPCBinaryClassification(Task):
    out_dir = Parameter("./prediction_binary/")
    graphIndex = Parameter("")
    gram_path = Parameter("")
    label_path = Parameter("")

    def __init__(self, identifier, x, y, C, h, D,
                 properties, train_index, test_index,
                 probability=True):
        self.identifier = identifier
        self.x = x
        self.y = y
        self.C = C
        self.h = h
        self.D = D
        self.properties = properties
        self.train_index = train_index
        self.test_index = test_index
        self.probability = probability

    def require(self):
        return [
            LoadJsonTask(
                self.graphIndex.value
            ),
            LoadJsonTask(
                self.gram_path.value % (self.h, self.D)
            ),
            LoadJsonTask(
                self.label_path.value
            )
        ]

    def __taskid__(self):
        prop = 'overall'
        if len(self.properties) == 1:
            prop = self.properties[0]
        return "BinaryPrediction_%s_%s_x_%d_y_%d_h_%d_D_%d" % (
            self.identifier, prop, self.x, self.y, self.h, self.D
        )

    def output(self):
        return LocalTarget(
            self.out_dir.value + self.__taskid__()+".json",
            service=JsonService
        )

    def run(self):
        with self.input()[0] as i:
            graphIndex = i.query()['index']

        with self.input()[1] as i:
            X = np.array(i.query())

        with self.input()[2] as i:
            all_label = i.query()

        print("Finished loading")

        train_index, y_train = create_index(
            self.train_index, graphIndex, all_label['index'], self.properties
        )

        test_index, y_test_index = create_index(
            self.test_index, graphIndex, all_label['index'], self.properties
        )

        y = np.array(all_label['matrix'])
        n = len(all_label['tools'])
        y = y[:, index(self.x, self.y, n) - n]

        y_train = y[y_train]
        y_test = y[y_test_index]
        del y

        X_train = X[train_index, :][:, train_index]

        m = y_train.nonzero()[0]
        y_train = y_train[m]
        y_train[np.where(y_train == -1)] = 0
        X_train = X_train[m, :][:, m]

        X_test = X[self.test_index, :][:, train_index]
        X_test = X_test[:, m]

        mt = y_test.nonzero()[0]
        y_test = y_test[mt]
        y_test[np.where(y_test == -1)] = 0
        X_p_test = X[test_index, :][mt, :][:, train_index][:, m]
        del X

        print("Finished slicing")

        clf = SVC(
            kernel='precomputed', probability=self.probability
        )
        params = {
            'C': self.C
        }

        clf = GridSearchCV(
            clf, params, 'accuracy', n_jobs=-1, iid=False, cv=10
        )
        print("Start training...")
        clf.fit(X_train, y_train)

        result = {
            'params': {
                'id': self.identifier,
                'C': self.C,
                'h': self.h,
                'D': self.D,
                'first_tool': self.x,
                'second_tool': self.y
            },
            'best_estimator': {
                'C': float(clf.best_params_['C']),
                'accuracy': float(clf.best_score_),
                'test_score': float(clf.score(X_p_test, y_test))
            }
        }

        print("Start prediction...")

        if self.probability:
            y = clf.predict_proba(X_test)[:, 1]
        else:
            y = clf.predict(X_test)

        result['prediction'] = y.tolist()

        with self.output() as o:
            o.emit(result)


def predicted_rankings(P):
    votings = []

    for (x, y), p in P.items():
        if len(votings) == 0:
            votings = [None]*p.shape[0]
        for i in range(p.shape[0]):
            if votings[i] is None:
                votings[i] = {}
            if x not in votings[i]:
                votings[i][x] = 0.0
            if y not in votings[i]:
                votings[i][y] = 0.0
            votings[i][x] += p[i]
            votings[i][y] += (1 - p[i])

    rankings = []
    for V in votings:
        rankings.append(
            [x for x, v in sorted(
                list(V.items()), key=lambda X: X[1], reverse=True
            )]
        )

    return rankings


class RPCRankPredictionEvaluation(Task):
    out_dir = Parameter("./ranking/")
    rank_path = Parameter('')
    label_path = Parameter("")
    graphIndex_path = Parameter("")

    def __init__(self, identifier, tool_count, C, h, D, measures,
                 properties, train_index, test_index,
                 probability=True):
        self.identifier = identifier
        self.tool_count = tool_count
        self.C = C
        self.h = h
        self.D = D
        self.properties = properties
        self.train_index = train_index
        self.test_index = test_index
        self.measures = measures
        self.probability = probability

    def require(self):
        base = [
            LoadJsonTask(
                self.rank_path.value
            ),
            LoadJsonTask(
                self.label_path.value
            ),
            LoadJsonTask(
                self.graphIndex_path.value
            )
        ]

        for x in range(self.tool_count):
            for y in range(x+1, self.tool_count):
                base.append(
                    RPCBinaryClassification(
                        self.identifier,
                        x,
                        y,
                        self.C,
                        self.h,
                        self.D,
                        self.properties,
                        self.train_index,
                        self.test_index,
                        self.probability
                    )
                )

        return base

    def __taskid__(self):
        prop = 'overall'
        if len(self.properties) == 1:
            prop = self.properties[0]
        return "RankEvaluation_%s_%s_h_%d_D_%d" % (
            self.identifier, prop, self.h, self.D
        )

    def output(self):
        return LocalTarget(
            self.out_dir.value + self.__taskid__()+".json",
            service=JsonService
        )

    def run(self):
        sub_scores = []

        prediction = {}
        for i in range(3, len(self.input())):
            with self.input()[i] as inp:
                P = inp.query()
            x = P['params']['first_tool']
            y = P['params']['second_tool']
            sub_scores.append(
                [x, y, P['best_estimator']['C'],
                 P['best_estimator']['accuracy'],
                 P['best_estimator']['test_score']
                 ]
            )
            prediction[(x, y)] = np.array(P['prediction'])

        rankings = predicted_rankings(prediction)

        with self.input()[1] as i:
            tools = i.query()['tools']

        rankings = [[tools[r] for r in R] for R in rankings]

        print("Calculated entries...")

        scores = {}
        for k in self.measures:
            scores[k] = select_score(k, {}, {})

        with self.input()[0] as i:
            expected = i.query()

        with self.input()[2] as i:
            graphIndex = i.query()['index']
        rev_index = {i: k for k, i in graphIndex.items()}

        scorings = {}

        seen = set([])

        for i, t in enumerate(self.test_index):
            if t in seen:
                continue
            seen.add(t)
            g_name = rev_index[t]
            for prop in self.properties:
                if prop not in expected[g_name]:
                    continue
                expected_ranking = expected[g_name][prop]
                ranking = rankings[i]

                if len(expected_ranking) == 0:
                    continue
                for k, score in scores.items():
                    if k not in scorings:
                        scorings[k] = []
                    scorings[k].append(
                            score(ranking, expected_ranking)
                    )

        for k, scores in list(scorings.items()):
            scorings[k] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }

        with self.output() as o:
            o.emit({
                'params': {
                    'identifier': self.identifier,
                    'tool_count': self.tool_count,
                    'C': self.C,
                    'h': self.h,
                    'D': self.D,
                    'properties': self.properties
                },
                'binary_scores': sub_scores,
                'rank_scores': scorings
            })


class RPCRankPredictionEvaluationCV(Task):
    out_dir = Parameter("./ranking/")

    def __init__(self, identifier, tool_count, C, h, D, measures,
                 properties, train_index, probability=True):
        self.identifier = identifier
        self.tool_count = tool_count
        self.C = C
        self.h = h
        self.D = D
        self.properties = properties
        self.train_index = train_index
        self.measures = measures
        self.probability = probability

    def require(self):
        fold = KFold(10, True, 42)
        base = []
        train = np.array(self.train_index)
        for i, (train_index, test_index) in enumerate(fold.split(train)):
            base.append(
                RPCRankPredictionEvaluation(
                        self.identifier + ('_%d' % i),
                        self.tool_count,
                        self.C,
                        self.h,
                        self.D,
                        self.measures,
                        self.properties,
                        train[train_index].tolist(),
                        train[test_index].tolist(),
                        self.probability
                    )
            )

        return base

    def __taskid__(self):
        prop = 'overall'
        if len(self.properties) == 1:
            prop = self.properties[0]
        return "RankEvaluationCV_%s_%s_h_%d_D_%d" % (
            self.identifier, prop, self.h, self.D
        )

    def output(self):
        return LocalTarget(
            self.out_dir.value + self.__taskid__()+".json",
            service=JsonService
        )

    def run(self):
        scorings = {}

        for inp in self.input():
            with inp as i:
                rpc = i.query()
            for score, value in rpc['rank_scores'].items():
                if score not in scorings:
                    scorings[score] = []
                scorings[score].append(value['mean'])

        for score, values in list(scorings.items()):
            scorings[score] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        with self.output() as o:
            o.emit({
                'params': {
                    'identifier': self.identifier,
                    'tool_count': self.tool_count,
                    'C': self.C,
                    'h': self.h,
                    'D': self.D,
                    'properties': self.properties
                },
                'scores': scorings
            })
