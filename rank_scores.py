

def select_score(type_id, map_graph_to_labels, map_graph_to_times):
    if type_id == 'accuracy':
        return accuracy_score
    elif type_id == 'spearmann':
        return spearmann_score
    elif type_id == 'spearmann_lambda':
        return spearmann_lambda_score
    elif type_id == 'kendall_tau':
        return kendall_tau_score
    elif type_id == 'inv_kendall_tau':
        return inv_kendall_tau_score
    elif type_id == 'svcomp':
        return make_svcomp_score(map_graph_to_labels)
    elif type_id == 'correct':
        return make_correct_score(map_graph_to_labels)
    elif type_id == 'time':
        return make_time_score(map_graph_to_labels, map_graph_to_times)
    elif type_id == 'time_count':
        return make_time_count_score(map_graph_to_labels, map_graph_to_times)
    elif type_id == 'inv_time':
        return make_inv_time_score(map_graph_to_labels, map_graph_to_times)
    else:
        raise ValueError('Unknown score %s' % type_id)


def accuracy_score(prediction_ranking, expected_ranking, graph=None):
    if prediction_ranking[0] == expected_ranking[0]:
        return 1
    return 0


def norm_rank(A, B):
    return [x for x in A if x in B], [y for y in B if y in A]


def spearmann_score(prediction_ranking, expected_ranking, graph=None):
    prediction_ranking, expected_ranking = norm_rank(prediction_ranking, expected_ranking)
    pr = {t: i for i, t in enumerate(prediction_ranking)}
    er = {t: i for i, t in enumerate(expected_ranking)}
    rg = 0.0
    n = len(prediction_ranking)

    for k in er:
        rg += (er[k] - pr[k])**2

    return 1 - (6*rg / (n * (n**2 - 1)))


def spearmann_lambda_score(prediction_ranking, expected_ranking, graph=None):
    if spearmann_score(prediction_ranking, expected_ranking, graph) >= 0.5:
        return 1
    return 0


def kendall_tau_distance(prediction_ranking, expected_ranking):
    pr = {t: i for i, t in enumerate(prediction_ranking)}

    dis = 0

    for i in range(len(expected_ranking)-1):
        for j in range(1, len(expected_ranking)):
            t1 = expected_ranking[i]
            t2 = expected_ranking[j]
            if pr[t1] > pr[t2]:
                dis += 1

    return dis


def kendall_tau_score(prediction_ranking, expected_ranking, graph=None):
    prediction_ranking, expected_ranking = norm_rank(prediction_ranking, expected_ranking)

    dis = kendall_tau_distance(prediction_ranking, expected_ranking)
    n = len(prediction_ranking)

    return 4 * dis / (n * (n - 1))


def inv_kendall_tau_score(prediction_ranking, expected_ranking, graph=None):
    return 1 - kendall_tau_score(prediction_ranking, expected_ranking, graph)


def svcomp_scoring(prediction_solve, expected_solve):
    if prediction_solve:
        return 1
    elif not prediction_solve and expected_solve:
        return -1
    else:
        return 0


def make_svcomp_score(map_graph_to_labels):

    def svcomp_score(prediction_ranking, expected_ranking, graph):
        if graph is None:
            raise ValueError('Need graph for calculation')

        label = map_graph_to_labels[graph]
        prediction_solve = label[prediction_ranking[0]]['solve'] == 'correct'
        expected_solve = label[expected_ranking[0]]['solve'] == 'correct'

        return svcomp_scoring(prediction_solve, expected_solve)

    return svcomp_score


def correct_scoring(prediction_solve, expected_solve):
    if prediction_solve or not expected_solve:
        return 1
    return 0


def make_correct_score(map_graph_to_labels):

    def correct_score(prediction_ranking, expected_ranking, graph):
        if graph is None:
            raise ValueError('Need graph for calculation')

        label = map_graph_to_labels[graph]
        prediction_solve = label[prediction_ranking[0]]['solve'] == 'correct'
        expected_solve = label[expected_ranking[0]]['solve'] == 'correct'

        return correct_scoring(prediction_solve, expected_solve)

    return correct_score


def make_time_score(map_graph_to_labels, map_graph_to_times):

    def time_score(prediction_ranking, expected_ranking, graph):
        if graph is None:
            raise ValueError('Need graph for calculation')

        label = map_graph_to_labels[graph]
        time = map_graph_to_times[graph]

        prediction = prediction_ranking[0]

        if 'prediction' in map_graph_to_times:
            prediction_time = map_graph_to_times['prediction']
        else:
            prediction_time = 0.0

        prediction_time += time + label[prediction]['time']
        full_time = sum([v['time'] for v in label.values()])
        return prediction_time / full_time

    return time_score


def make_time_count_score(map_graph_to_labels, map_graph_to_times):

    def time_score(prediction_ranking, expected_ranking, graph):
        if graph is None:
            raise ValueError('Need graph for calculation')

        label = map_graph_to_labels[graph]
        time = map_graph_to_times[graph]

        prediction = prediction_ranking[0]
        if 'prediction' in map_graph_to_times:
            prediction_time = map_graph_to_times['prediction']
        else:
            prediction_time = 0.0

        prediction_time += time + label[prediction]['time']
        full_time = sum([v['time'] for v in label.values()])
        return 1 if prediction_time < full_time else 0

    return time_score


def make_inv_time_score(map_graph_to_labels, map_graph_to_times):

    def inv_time_score(prediction_ranking, expected_ranking, graph):
        score = make_time_score(map_graph_to_labels, map_graph_to_times)
        return 1 - score(prediction_ranking, expected_ranking)

    return inv_time_score
