from Utils import *
from scipy.optimize import minimize


def calculate_resps_probabilities(params, resps, rewards):
    alpha_pos = params[0]
    alpha_neg = params[1]
    alpha_unchosen = params[2]
    beta = params[3]

    Q = np.array([0.5, 0.5])
    probabilities = np.zeros(len(resps))
    for t, (resp, reward) in enumerate(zip(resps, rewards)):
        choice_prob = softmax_func(Q, beta)
        probabilities[t] = choice_prob[resp]
        
        prediction_error = rewards[t] - Q[resp]    
        if prediction_error > 0:
            Q[resp] = Q[resp] + alpha_pos * prediction_error
        else: 
            Q[resp] = Q[resp] + alpha_neg * prediction_error
        
        unchosen_option = 1 - resp
        Q[unchosen_option] = Q[unchosen_option] - alpha_unchosen * prediction_error
    return probabilities

def forward_LL(params, resps, rewards):
    probabilities = calculate_resps_probabilities(params, resps, rewards)
    return -np.sum(protected_log(probabilities))


def model_subj_conf_dist(params, confs, resp_probs):
    lbound, scale = params[0], params[1]
    hbound = scale * (5 - lbound) + lbound
    model_confs = linear_transform_on_array(resp_probs, [0, 1], [lbound, hbound])
    return mean_square_distance(confs, model_confs)

def fit_n_times(n_fitting, params_ranges, func, func_args):
    results = [minimize(func, x0=[np.random.uniform(*r) for r in params_ranges], args=func_args, bounds=params_ranges) for _ in range(n_fitting)]
    best_result = min(results, key=lambda x: x.fun if x.success else np.inf)
    return best_result.x, [result.fun for result in results if result.success]


def forward_fitting(dict_input):
    resps, rewards, confs, n_fitting, params_ranges, scale = dict_input.values()
    model_param_ranges, bound_ranges = params_ranges[:4], params_ranges[4:]

    fit_res, neg_LL_seq = fit_n_times(n_fitting, model_param_ranges, forward_LL, (resps, rewards))
    alpha, beta = fit_res[:2]

    resp_probs = calculate_resps_probabilities(fit_res, resps, rewards)
    fit_res_conf, conf_dist_seq = fit_n_times(n_fitting, bound_ranges, model_subj_conf_dist, (confs, resp_probs))

    fit_res_final = (*fit_res, *fit_res_conf)
    return fit_res_final, min(neg_LL_seq), min(conf_dist_seq)



def simulate_agent(params, trials_info):
    alpha_pos = params[0]
    alpha_neg = params[1]
    alpha_unchosen = params[2]
    beta = params[3]

    lbound = params[4]
    hbound = params[5]

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
        
        if prediction_error > 0:
            Q[resps[t]] = Q[resps[t]] + alpha_pos * prediction_error
        else: 
            Q[resps[t]] = Q[resps[t]] + alpha_neg * prediction_error
        
        unchosen_option = 1 - resps[t]
        Q[unchosen_option] = Q[unchosen_option] - alpha_unchosen * prediction_error

    confs = linear_transform_on_array(chosen_probs, [0, 1], [lbound, hbound])
    return resps, rewards, acc, confs, Q_list


def simulate_with_params(params_df, data, subjects, n= 100):
    df_lists = []
    for i, row in params_df.iterrows():
        params = [row["alpha_pos"],row["alpha_neg"], row["alpha_unchosen"], row["beta"], row["lbound"], row["hbound"]]
        subject = int(row["subject"])
        trials_info = get_subject_task(data, subject)
        for run in range(n):
            resps, rewards, acc, confs, Q_list = simulate_agent(params, trials_info)
            subject_list = [subject] * (len(resps))
            run_list = [run] * (len(resps))
            df = pd.DataFrame(zip(subject_list, run_list,resps, rewards, acc, confs, Q_list[:, 0], Q_list[:, 1]),
                          columns=["sub", "run", "resp", "reward", "cor", "cj", "Q1", "Q2"])
            df_lists.append(df)
    return pd.concat(df_lists, axis=0, ignore_index=True)

def simulate_with_params_one(params_df, data, subjects):
    df_lists = []
    for i, row in params_df.iterrows():
        params = [row["alpha_pos"],row["alpha_neg"], row["alpha_unchosen"], row["beta"], row["lbound"], row["hbound"]]
        subject = int(row["subject"])
        trials_info = get_subject_task(data, subject)
        for run in range(1):
            resps, rewards, acc, confs, Q_list = simulate_agent(params, trials_info)
            subject_list = [subject] * (len(resps))
            run_list = [run] * (len(resps))
            df = pd.DataFrame(zip(subject_list, run_list,resps, rewards, acc, confs, Q_list[:, 0], Q_list[:, 1]),
                          columns=["sub", "run", "resp", "reward", "cor", "cj", "Q1", "Q2"])
        df_lists.append(df)
    return pd.concat(df_lists, axis=0, ignore_index=True)

