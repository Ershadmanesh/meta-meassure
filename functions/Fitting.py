from Utility import *
from multiprocessing import Pool
from scipy.optimize import minimize



def fit_for_one_subject(params_ranges,n_fitting, agent, data, fitting_func):
  d = get_task_info(data, agent)
  d["n_fitting"] = n_fitting
  d["params_ranges"] = params_ranges
  result = fitting_func(d)
  return [result]

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


def minimize_brier(dict_input):
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
