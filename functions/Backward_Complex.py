from Utils import *
from scipy.optimize import minimize
from Forward import forward_LL

def calculate_choice_probabilities(Q, beta, resps, rewards, alpha_pos, alpha_neg, alpha_unchosen):
    """Calculate choice probabilities and update Q values based on responses and rewards."""
    task_length = len(resps)
    probabilities = np.zeros(task_length)
    for t in range(task_length):
        choices_probability = softmax_func(Q, beta)
        probabilities[t] = choices_probability[resps[t]]
        prediction_error = rewards[t] - Q[resps[t]]
        if prediction_error > 0:
            Q[resps[t]] = Q[resps[t]] + alpha_pos * prediction_error
        else: 
            Q[resps[t]] = Q[resps[t]] + alpha_neg * prediction_error
        
        unchosen_option = 1 - resps[t]
        Q[unchosen_option] = Q[unchosen_option] - alpha_unchosen * prediction_error
    return probabilities

def backward_distance(params, resps, rewards, confs, scale):
    """Calculate mean squared distance between experimental and model confidence levels."""
    alpha_pos, alpha_neg, alpha_unchosen, beta= params[:4]
    Q = np.array([.5, .5])
    
    probabilities = calculate_choice_probabilities(Q, beta, resps, rewards, alpha_pos, alpha_neg, alpha_unchosen)
    
    if scale:
        lbound, hbound = params[2], params[3] * (5 - params[2]) + params[2]
        model_confs = linear_transform_on_array(probabilities, [0, 1], [lbound, hbound])
    else:
        model_confs = linear_transform_on_array(probabilities, [0, 1], [1, 5])
    
    return mean_square_distance(confs, model_confs)

def fit_model_backward(dict_input):
    """Fit model by minimizing the backward distance and calculate negative log-likelihood."""
    resps, rewards, confs, scale = dict_input["resp"], dict_input["reward"], dict_input["cj"], dict_input["scale"]
    n_fitting, params_ranges = dict_input["n_fitting"], dict_input["params_ranges"]
    
    best_fit, min_dist, neg_LL = None, np.inf, None
    
    for _ in range(n_fitting):
        initial_params = [np.random.uniform(*range_) for range_ in params_ranges]
        result = minimize(backward_distance, initial_params, args=(resps, rewards, confs, scale), bounds=params_ranges)
        
        if result.success and result.fun < min_dist:
            min_dist, best_fit = result.fun, result.x
            neg_LL = forward_LL(best_fit, resps, rewards)
    
    return best_fit, neg_LL, min_dist