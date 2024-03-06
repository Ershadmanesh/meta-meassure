from Utils import *
from multiprocessing import Pool
from scipy.optimize import minimize

def prepare_input_dict(data, subject, n_fitting, params_ranges, scale = True):
    """Prepare the input dictionary for a single subject."""
    d = get_task_info(data, subject)
    d.update({"n_fitting": n_fitting, "params_ranges": params_ranges, "scale": scale})
    return d

def minimize_brier_for_subject(dict_input):
    """Minimize the Brier score for a single subject."""
    confs, cors = dict_input["confs"], dict_input["cor"]
    n_fitting = 60 
    min_brier = np.inf
    model_param_ranges = [[0,1], [.01,1]]
    for j in range(n_fitting):
        params0 = np.zeros(len(model_param_ranges))
    for p in range(len(model_param_ranges)):
        params0[p] = (np.random.uniform(*model_param_ranges[p]))
    fitting_result = minimize(group_brier_scaled,x0= params0, args=(cors, confs), bounds=model_param_ranges)
    if fitting_result.success:
        if fitting_result.fun < min_brier:
            min_brier = fitting_result.fun
            fit_res = fitting_result.x
    return fit_res, min_brier

def fit_for_all_subjects_parallel(params_ranges, n_threads, n_fitting, subjects, data, fitting_func, scale):
    """Fit models for all subjects in parallel."""
    input_dicts = [prepare_input_dict(data, sub, n_fitting, params_ranges, scale) for sub in subjects]
    with Pool(n_threads) as pool:
        results = pool.map(fitting_func, input_dicts)
    return results

def minimize_brier(dict_input):
    """Minimize the Brier score for a single subject."""
    confs = dict_input["confs"]
    cors = dict_input["cor"]
    n_fitting = 60 
    min_brier = np.inf
    model_param_ranges = [[0,1], [.01,1]]
    for j in range(n_fitting):
      params0 = np.zeros(len(model_param_ranges))
    for p in range(len(model_param_ranges)):
      params0[p] = (np.random.uniform(*model_param_ranges[p]))
    fitting_result = minimize(group_brier_scaled,x0= params0, args=(cors, confs), bounds=model_param_ranges)
    if fitting_result.success:
      if fitting_result.fun < min_brier:
        min_brier = fitting_result.fun
        fit_res = fitting_result.x
    return fit_res, min_brier

def fit_brier(subjects, data, n_threads):
  input_dict = []

  for sub in subjects:
    one_subject_data = data[data["sub"] == sub]
    cor = np.array(one_subject_data["cor"])
    confs =  np.array(one_subject_data["cj"])
    input_dict.append({"cor" : cor, "confs": confs})

  with Pool(n_threads) as p:
    result = p.map(minimize_brier, input_dict)
  return result
