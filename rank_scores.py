

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
    elif type_id == 'correct_count':
        return make_correct_count_score(map_graph_to_labels)
    elif type_id == 'time':
        return make_time_score(map_graph_to_labels, map_graph_to_times)
    elif type_id == 'time_count':
        return make_time_count_score(map_graph_to_labels, map_graph_to_times)
    elif type_id == 'inv_time':
        return make_inv_time_score(map_graph_to_labels, map_graph_to_times)
    elif type_id == "fail_fast":
        return make_fail_fast_count(map_graph_to_labels)
    else:
        raise ValueError('Unknown score %s' % type_id)


def accuracy_score(prediction_ranking, expected_ranking, graph=None):
    expected_ranking, prediction_ranking = restricted_tie_break(expected_ranking, prediction_ranking)
    if prediction_ranking[0] == expected_ranking[0]:
        return 1
    return 0


def _is_list(obj):
    if isinstance(obj, str):
        return False
    try:
        _ = (e for e in obj)
        return True
    except TypeError:
        return False


# Assume B has no ties
def restricted_tie_break(A, B):
    b = {t: i for i, t in enumerate(B)}

    R = []
    for a in A:
        if _is_list(a):
            a = sorted(list(a), key=lambda x: b[x])
            R.extend(a)
        else:
            R.append(a)

    return R, B


def spearmann_score(prediction_ranking, expected_ranking, graph=None):
    expected_ranking, prediction_ranking = restricted_tie_break(expected_ranking, prediction_ranking)
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


def _kendall_rank_pairs(ranking):
    for i in range(len(ranking)-1):
        for j in range(i + 1, len(ranking)):
            a = ranking[i]
            b = ranking[j]

            if _is_list(a):
                for _a in a:
                    yield (_a, b)
            elif _is_list(b):
                for _b in b:
                    yield (a, _b)
            else:
                yield (a, b)


def kendall_tau_score(prediction_ranking, expected_ranking, graph=None):
    count = 0
    n_c = 0
    n_d = 0
    pr = {t: i for i, t in enumerate(prediction_ranking)}

    for (a, b) in _kendall_rank_pairs(expected_ranking):
        if pr[a] < pr[b]:
            n_c += 1
        else:
            n_d += 1
        count += 1

    if count == 0:
        return 1

    return (n_c - n_d) / count


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


def make_correct_count_score(map_graph_to_labels):

    def correct_score(prediction_ranking, expected_ranking, graph):
        if graph is None:
            raise ValueError('Need graph for calculation')

        label = map_graph_to_labels[graph]
        prediction_solve = label[prediction_ranking[0]]['solve'] == 'correct'

        return 1 if prediction_solve else 0

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

        print('prediction_time: %s (%s)\n time: %s (%s)\n label_time: %s (%s)'
              % (str(prediction_time), str(type(prediction_time)),
                 str(time), str(type(time)),
                 str(label[prediction]['time']),
                 str(type(label[prediction]['time']))))

        prediction_time += time + label[prediction]['time']
        full_time = sum([v['time'] for v in label.values()])
        return 1 if prediction_time < full_time else 0

    return time_score


def make_inv_time_score(map_graph_to_labels, map_graph_to_times):

    def inv_time_score(prediction_ranking, expected_ranking, graph):
        score = make_time_score(map_graph_to_labels, map_graph_to_times)
        return 1 - score(prediction_ranking, expected_ranking)

    return inv_time_score


def make_fail_fast_count(map_graph_to_labels):

    def fail_fast_count(prediction_ranking, expected_ranking, graph):
        if graph is None:
            raise ValueError('Need graph for calculation')

        label = map_graph_to_labels[graph]

        pL = label[prediction_ranking[0]]
        eL = label[prediction_ranking[1]]

        if pL['solve'] == 'false' and eL['solve'] == 'false':
            if float(pL['time']) <= eL['time']:
                return 1

        return 0

    return fail_fast_count
