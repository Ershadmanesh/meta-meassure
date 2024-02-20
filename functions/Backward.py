from Utility import *
from scipy.optimize import minimize
from Forward import forward_LL


def backward_distance(params, resps, rewards, confs):
    alpha = params[0]
    beta = params[1]

    lbound = params[2]
    hbound = params[3] * (5 - lbound) + lbound

    task_length = len(resps)
    resps_probabilities = np.zeros(task_length)
    Q = np.array([.5, .5])
    for t in range(task_length):
        choices_probability = softmax_func(Q, beta)
        resps_probabilities[t] = choices_probability[resps[t]]
        prediction_error = rewards[t] - Q[resps[t]]
        Q[resps[t]] = Q[resps[t]] + alpha * prediction_error

    model_confs = linear_transform_on_array(resps_probabilities, [0, 1], [lbound, hbound])
    dist = mean_square_distance(confs, model_confs)
    return dist


def backward_fitting(dict_input):
    resps = dict_input["resps"]
    rewards = dict_input["rewards"]
    confs = dict_input["confs"]
    n_fitting = dict_input["n_fitting"]
    params_ranges = dict_input["params_ranges"]
    min_dist = np.inf
    neg_LL = 0
    dist_seq = np.zeros(n_fitting)
    neg_LL_seq = np.zeros(n_fitting)
    for j in range(n_fitting):
        params0 = np.zeros(len(params_ranges))
        for p in range(len(params_ranges)):
            params0[p] = (np.random.uniform(*params_ranges[p]))
        fitting_result = minimize(backward_distance, x0=params0, args=(resps, rewards, confs), bounds=params_ranges)
        if fitting_result.success:
            if fitting_result.fun < min_dist:
                min_dist = fitting_result.fun
                fit_res = fitting_result.x
                neg_LL = forward_LL(fit_res, resps, rewards)
        dist_seq[j] = min_dist
        neg_LL_seq[j] = neg_LL
    return fit_res, neg_LL_seq, dist_seq

def backward_distance_unscaled(params, resps, rewards, confs):
    alpha = params[0]
    beta = params[1]

    task_length = len(resps)
    resps_probabilities = np.zeros(task_length)
    Q = np.array([.5, .5])
    for t in range(task_length):
        choices_probability = softmax_func(Q, beta)
        resps_probabilities[t] = choices_probability[resps[t]]
        prediction_error = rewards[t] - Q[resps[t]]
        Q[resps[t]] = Q[resps[t]] + alpha * prediction_error

    model_confs = linear_transform_on_array(resps_probabilities, [0, 1], [1, 5])
    dist = mean_square_distance(confs, model_confs)
    return dist


def backward_fitting_unscaled(dict_input):
    resps = dict_input["resps"]
    rewards = dict_input["rewards"]
    confs = dict_input["confs"]
    n_fitting = dict_input["n_fitting"]
    params_ranges = dict_input["params_ranges"]
    min_dist = np.inf
    neg_LL = 0
    dist_seq = np.zeros(n_fitting)
    neg_LL_seq = np.zeros(n_fitting)
    for j in range(n_fitting):
        params0 = np.zeros(len(params_ranges))
        for p in range(len(params_ranges)):
            params0[p] = (np.random.uniform(*params_ranges[p]))
        fitting_result = minimize(backward_distance_unscaled, x0=params0, args=(resps, rewards, confs), bounds=params_ranges)
        if fitting_result.success:
            if fitting_result.fun < min_dist:
                min_dist = fitting_result.fun
                fit_res = fitting_result.x
                neg_LL = forward_LL(fit_res, resps, rewards)
        dist_seq[j] = min_dist
        neg_LL_seq[j] = neg_LL
    return fit_res, neg_LL_seq, dist_seq

