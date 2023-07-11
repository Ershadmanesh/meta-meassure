import numpy as np
import pandas as pd
import sys


def protected_log(x):
    'Return log of x protected against giving -inf for very small values of x.'

    return np.log((1e-200 / 2) + (1 - 1e-200) * x)


def softmax_func(Q, beta):
    log_max_float = np.log(sys.float_info.max / 2.1)
    """
    Array based calculation of softmax probabilities for binary choices.
    """
    P = np.zeros(2)
    try:
        # Code that may cause an overflow warning
        TdQ = -beta * (Q[0] - Q[1])
    except RuntimeWarning:
        # Code to handle the warning
        print(beta, Q[0], Q[1])
        TdQ = 0

    if TdQ > log_max_float:  # Protection against overflow in exponential.
        TdQ = log_max_float

    P[0] = 1. / (1. + np.exp(TdQ))

    P[1] = 1. - P[0]

    return P


def get_task_info(data, subject):
    one_subject_data = data[data["sub"] == subject]
    rewards = np.array(one_subject_data["reward"])
    resps = np.array(one_subject_data["resp"])
    confs = np.array(one_subject_data["cj"])
    return {"resps": resps, "rewards": rewards, "confs": confs}


def linear_transform_on_array(input_array, domain_range, codomain_range):
    xl, xu = domain_range
    yl, yu = codomain_range
    if xl != xu:
        y = ((yu - yl) / (xu - xl)) * (input_array - xl) + yl
    else:
        y = np.ones_like(input_array) * yl
    return y


def mean_square_distance(x, y):
    return np.mean((x - y) ** 2)


def make_df_after_fitting(fitting_res, paramter_names, subjects):
    n_params = len(paramter_names)
    df_data = np.zeros((len(fitting_res), n_params + 3))
    df_names = ["subject"]
    for name in paramter_names:
        df_names.append(name)

    df_names.append("LL")
    df_names.append("Confidence Distance")
    count = 0

    for res in fitting_res:
        df_data[count, 0] = subjects[count]
        fitted_param = res[0]
        for i in range(len(paramter_names)):
            df_data[count, i + 1] = fitted_param[i]
        df_data[count, len(paramter_names) + 1] = res[1][-1]
        df_data[count, len(paramter_names) + 2] = res[2][-1]
        count += 1
    return pd.DataFrame(df_data, columns=df_names)


def merge_result_df(dataframes, labels, gap, subjects):
    subj_to_ind = {}
    subj_to_ind_low = {}
    idx = 1
    low_idx = 1
    for subject in subjects:
        subj_to_ind[subject] = idx
        subj_to_ind_low[subject] = low_idx
        idx += 1
        low_idx += 3

    count = 0
    for df, label in zip(dataframes, labels):
        df["model"] = label
        df["x_idx"] = df.apply(lambda x: subj_to_ind[x["subject"]] + count * gap, axis=1)
        count += 1
    return pd.concat(dataframes, ignore_index=True)


def get_subject_task(data, subject):
    one_subject_data = data[data["sub"] == subject]
    # extracting block rewards
    blocks = {}
    block_idx = 0
    block_trials = []
    for i, row in one_subject_data.iterrows():
        block_trials.append((row["cresp"], row["Lreward"] / 100, row["Hreward"] / 100, row["condition"]))
        if row["trial_rev"] == 0:
            blocks[block_idx] = block_trials
            block_idx = block_idx + 1
            block_trials = []
    trials_info = []
    for block, trials in blocks.items():
        trials_info.extend(trials)
    return trials_info


def group_qsr(ser):
    cor = ser["cor"]
    c = linear_transform_on_array(ser["cj"], [1, 5], [0, 1])
    return 1- np.mean(np.power(cor - c,2))

