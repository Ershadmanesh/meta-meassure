
import numpy as np

from Fitting import fit_for_one_subject, minimize_brier
from Backward import backward_fitting
from Forward import simulate_with_params
from Utility import *

def noisy_agent(vr, n_reps, n_fitting, params_ranges, data, subject_id, subjects):
  
  lsyn_backward_means = np.zeros(n_reps)
  lsyn_metai = np.zeros(n_reps)
  lsyn_qsr = np.zeros(n_reps)
  lsyn_qsr_scaled = np.zeros(n_reps)
  selected_sub_data = data[data['sub'] == subject_id]    

  for i in range(n_reps):
    low_syn_df = selected_sub_data.copy()
    cor_trials_ind = low_syn_df[low_syn_df['cor']== 1].index
    incor_trials_ind = low_syn_df[low_syn_df['cor']== 0].index
    low_syn_df.loc[cor_trials_ind, 'cj'] = 5 - np.abs(np.random.normal(0, vr, len(cor_trials_ind)))
    low_syn_df.loc[incor_trials_ind, 'cj'] = 1 + np.abs(np.random.normal(0, vr, len(incor_trials_ind)))
    low_syn_df.loc[low_syn_df['cj'] < 1, 'cj']['cj'] = 1
    low_syn_df.loc[low_syn_df['cj'] > 5, 'cj'] = 5
    res = fit_for_one_subject(params_ranges,n_fitting,subject_id, low_syn_df, backward_fitting)
    lsyn_backward_df = make_df_after_fitting(res, ["alpha", "beta", "lbound", "bound_range"], [subject_id])
    lsyn_backward_df["hbound"] = (5 - lsyn_backward_df["lbound"])*(lsyn_backward_df["bound_range"]) + lsyn_backward_df["lbound"]
    lsyn_backward_simulation_df = simulate_with_params(lsyn_backward_df, data,[subject_id])
    lsyn_backward_means[i] = lsyn_backward_simulation_df['cor'].mean()
    lsyn_metai[i] = calculate_metai(low_syn_df)
    lsyn_qsr[i] = group_qsr(low_syn_df)
    _, min_brier = minimize_brier({"confs": np.array(low_syn_df["cj"]), "cor" : np.array(low_syn_df["cor"])})
    lsyn_qsr_scaled[i] = 1 - min_brier
  return lsyn_backward_means, lsyn_metai, lsyn_qsr, lsyn_qsr_scaled