import numpy as np
import pandas as pd
import sys
import seaborn as sns


def protected_log(x):
    """Return log of x protected against giving -inf for very small values of x."""
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
    """Extracts task-specific information for a given subject."""
    one_subject_data = data[data["sub"] == subject]
    return {col: one_subject_data[col].to_numpy() for col in ["resp", "reward", "cj"]}


def linear_transform_on_array(input_array, domain_range, codomain_range):
    """Linearly transforms an array from one range to another."""
    xl, xu = domain_range
    yl, yu = codomain_range
    scale = (yu - yl) / (xu - xl) if xl != xu else 0
    return np.where(xl != xu, ((input_array - xl) * scale) + yl, yl)


def mean_square_distance(x, y):
    """Calculates the mean squared distance between two arrays."""
    return np.mean((x - y) ** 2)

def make_df_after_fitting(fitting_res, paramter_names, subjects):
    """Creates a DataFrame after model fitting."""
    df_data = {"subject": subjects}
    df_data.update({name: [res[0][i] for res in fitting_res] for i, name in enumerate(paramter_names)})
    df_data.update({"LL": [res[1] for res in fitting_res], "Confidence Distance": [res[2] for res in fitting_res]})
    return pd.DataFrame(df_data)


def merge_result_df(dataframes, labels, gap, subjects):
    """Merges multiple result DataFrames."""
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
    """Calculates a QSR metric."""
    c = linear_transform_on_array(ser["cj"], [1, 5], [0, 1])
    return 1 - np.mean(np.power(ser["cor"] - c, 2))

def group_brier_scaled(b, cor, conf):
    """Applies a scaled Brier score calculation for evaluating predictions."""
    lb, hb = linear_transform_on_array(np.array([b[0], b[1]]), [0, 1], [b[0], b[1] * (1 - b[0]) + b[0]])
    c = linear_transform_on_array(conf, [1, 5], [lb, hb])
    return np.mean(np.power(cor - c, 2))

# Load and preprocess data
def preprocess_data(file_path, chance_level_subjects, phase, condition):
    raw_data = pd.read_csv(file_path)
    data = raw_data[["sub", "condition", "cresp", "resp", "cor", "cj", "phase", "trial", "trial_rev", "reward", "Lreward", "Hreward"]]
    filtered_data = data[(data["phase"] == phase) & (data["condition"] ==condition) & (~data["sub"].isin(chance_level_subjects))]
    return filtered_data

