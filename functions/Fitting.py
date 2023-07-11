from Utility import *
from multiprocessing import Pool


def fit_for_all_subjects(params_ranges, n_threads, n_fitting, subjects, data, fitting_func):
    """
    Asynchronously Fitting the model to the experimental data
    """

    input_dicts = [get_task_info(data, sub) for sub in subjects]

    for d in input_dicts:
        d["n_fitting"] = n_fitting
        d["params_ranges"] = params_ranges

    with Pool(n_threads) as p:
        result = p.map(fitting_func, input_dicts)
    return result
