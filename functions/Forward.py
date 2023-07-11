from Utility import *
from scipy.optimize import minimize


def forward_LL(params, resps, rewards):
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

    neg_LL = -np.sum(protected_log(resps_probabilities))
    return neg_LL


def forward_probs(params, resps, rewards, *args):
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

    return resps_probabilities


def model_subj_conf_dist(params,confs, resp_probs):

    lbound = params[0]
    hbound = params[1]*(5 - lbound) + lbound

    chosen_probs = resp_probs.copy()

    model_confs = linear_transform_on_array(chosen_probs, [0, 1], [lbound, hbound])
    return mean_square_distance(confs, model_confs)


def fit_n_times(n_fitting, params_ranges, func, func_args):
    min_func = np.inf
    func_val_seq = np.zeros(n_fitting)
    fit_res = np.array([])
    for j in range(n_fitting):
        params0 = np.zeros(len(params_ranges))
        for p in range(len(params_ranges)):
            params0[p] = (np.random.uniform(*params_ranges[p]))
        fitting_result = minimize(func, x0=params0, args=func_args, bounds=params_ranges)
        if fitting_result.success:
            if fitting_result.fun < min_func:
                min_func = fitting_result.fun
                fit_res = fitting_result.x
        func_val_seq[j] = min_func
    return fit_res, func_val_seq


def forward_fitting(dict_input):
    """
    Fit data to forward model
    """

    resps = dict_input["resps"]
    rewards = dict_input["rewards"]
    confs = dict_input["confs"]
    n_fitting = dict_input["n_fitting"]
    params_ranges = dict_input["params_ranges"]

    # choice params
    model_param_ranges = params_ranges[:2]
    # cond params
    bound_ranges = params_ranges[2:]

    fit_res, neg_LL_seq = fit_n_times(n_fitting, model_param_ranges, forward_LL, (resps, rewards))

    alpha, beta = fit_res

    resp_probs = forward_probs(fit_res, resps, rewards)

    fit_res, conf_dist_seq = fit_n_times(n_fitting, bound_ranges, model_subj_conf_dist, (confs, resp_probs))

    lbound, bound_range = fit_res
    fit_res = (alpha, beta, lbound, bound_range)

    return fit_res, neg_LL_seq, conf_dist_seq


def simulate_agent(params, trials_info):
    alpha = params[0]
    beta = params[1]

    lbound = params[2]
    hbound = params[3]

    Q = np.array([0.5, 0.5])
    Q_list = np.zeros((len(trials_info), 2))
    rewards = []
    resps = []
    chosen_probs = np.zeros(len(trials_info))

    acc = np.zeros(len(trials_info))
    for t, trial in enumerate(trials_info):
        Q_list[t, :] = Q
        correct_resp = trial[0]
        low_reward = trial[1]
        high_reward = trial[2]
        choices_probability = softmax_func(Q, beta)
        resp = np.random.choice([0, 1], p=choices_probability)
        chosen_probs[t] = choices_probability[resp]
        if resp == correct_resp:
            reward = high_reward
            acc[t] = 1
        else:
            acc[t] = 0
            reward = low_reward
        resps.append(resp)
        rewards.append(reward)
        prediction_error = reward - Q[resp]
        Q[resps[t]] = Q[resps[t]] + alpha * prediction_error

    confs = linear_transform_on_array(chosen_probs, [0, 1], [lbound, hbound])
    return resps, rewards, acc, confs, Q_list


def simulate_with_params(params_df, data, subjects):
    df_lists = []
    for i, row in params_df.iterrows():
        params = [row["alpha"], row["beta"], row["lbound"], row["hbound"]]
        subject = int(row["subject"])
        trials_info = get_subject_task(data, subject)
        resps, rewards, acc, confs, Q_list = simulate_agent(params, trials_info)
        subject_list = [subject] * (len(resps))
        df = pd.DataFrame(zip(subject_list, resps, rewards, acc, confs, Q_list[:, 0], Q_list[:, 1]),
                          columns=["sub", "resp", "reward", "cor", "cj", "Q1", "Q2"])
        df_lists.append(df)
    return pd.concat(df_lists, axis=0, ignore_index=True)
