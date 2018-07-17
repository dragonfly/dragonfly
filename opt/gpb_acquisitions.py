"""
  Acquisition functions for Bayesian Optimisation.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import division

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=star-args

from builtins import zip
from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro
# Local
from utils.general_utils import solve_lower_triangular
from gp.gp_core import get_post_covar_from_raw_covar
from ed.domains import EuclideanDomain

def _optimise_acquisition(acq_fn, acq_optimiser, anc_data):
  """ All methods will just call this. """
  if anc_data.acq_opt_method in ['direct', 'pdoo']:
    acquisition = lambda x: acq_fn(x.reshape((1, -1)))
  else:
    acquisition = acq_fn
  _, opt_pt = acq_optimiser(acquisition, anc_data.max_evals)
  return opt_pt

def _get_halluc_points(_, halluc_pts):
  """ Re-formats halluc_pts if necessary. """
  if len(halluc_pts) > 0:
    return halluc_pts
  else:
    return halluc_pts

# Thompson sampling ---------------------------------------------------------------
def asy_ts(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via TS in the asyuential setting. """
  gp_sample = lambda x: gp.draw_samples(1, X_test=x, mean_vals=None, covar=None).ravel()
  return _optimise_acquisition(gp_sample, acq_optimiser, anc_data)

def asy_hts(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via TS using hallucinated observaitons in the asynchronus
      setting. """
  halluc_pts = _get_halluc_points(gp, anc_data.evals_in_progress)
  gp_sample = lambda x: gp.draw_samples_with_hallucinated_observations(1, x,
                                                                       halluc_pts).ravel()
  return _optimise_acquisition(gp_sample, acq_optimiser, anc_data)

def syn_ts(num_workers, gp, acq_optimiser, anc_data, **kwargs):
  """ Returns a batch of recommendations via TS in the synchronous setting. """
  recommendations = []
  for _ in range(num_workers):
    rec_j = asy_ts(gp, acq_optimiser, anc_data, **kwargs)
    recommendations.append(rec_j)
  return recommendations

# Add-UCB --------------------------------------------------------------------------
def _get_add_ucb_beta_th(dim, time_step):
  """ Computes the beta t for UCB based methods. """
  return np.sqrt(0.2 * dim * np.log(2 * dim * time_step + 1))

# TODO: add hallucinated observations for parallelisation.

# Set up an acquisition optimiser which can take domain as an argument.
def _get_add_acq_optimiser(group_bounds, acq_opt_method):
  """ Get an acquisition optimiser. """
  group_domain = EuclideanDomain(group_bounds)
  return lambda obj, max_evals: group_domain.maximise_obj(acq_opt_method,
                                                          obj, max_evals)

def _add_ucb(gp, add_kernel, mean_funcs, anc_data):
  """ Common functionality for Additive UCB acquisition under various settings.
  """
  # pylint: disable=undefined-loop-variable
  # prelims
  kernel_list = add_kernel.kernel_list
  groupings = add_kernel.groupings
  total_max_evals = anc_data.max_evals
  kern_scale = add_kernel.hyperparams['scale']
  domain_bounds = anc_data.domain_bounds
  X_train = np.array(gp.X)
  num_groups = len(kernel_list)
  if mean_funcs is None:
    mean_funcs = lambda x: np.array([0] * len(x))
  if not hasattr(mean_funcs, '__iter__'):
    mean_funcs = [mean_funcs] * num_groups
  group_points = []
  num_coordinates = 0
  anc_data.max_evals = total_max_evals//num_groups

  # Now loop through each group
  for group_j, kernel_j, mean_func_j in \
      zip(groupings, kernel_list, mean_funcs):
    # Using a python internal function in a loop is typically a bad idea. But we are
    # using this function only inside this loop, so it should be fine.
    def _add_ucb_acq_j(X_test_j):
      """ Acquisition for the jth group. """
      betath_j = _get_add_ucb_beta_th(len(group_j), anc_data.t)
      X_train_j = X_train[:, group_j]
      K_tetr_j = kern_scale * kernel_j(X_test_j, X_train_j)
      pred_mean_j = K_tetr_j.dot(gp.alpha) + mean_func_j(X_test_j)
      K_tete_j = kern_scale * kernel_j(X_test_j, X_test_j)
      V_j = solve_lower_triangular(gp.L, K_tetr_j.T)
      post_covar_j = K_tete_j - V_j.T.dot(V_j)
      post_covar_j = get_post_covar_from_raw_covar(post_covar_j, gp.noise_var,
                                                   gp.kernel.is_guaranteed_psd())
      post_std_j = np.sqrt(np.diag(post_covar_j))
      return pred_mean_j + betath_j * post_std_j
    # Objatin the jth group
    acq_optimiser_j = _get_add_acq_optimiser(domain_bounds[group_j],
                                             anc_data.acq_opt_method)
    point_j = _optimise_acquisition(_add_ucb_acq_j, acq_optimiser_j, anc_data)
    group_points.append(point_j)
    num_coordinates += len(point_j)

  # Now return
  anc_data.max_evals = total_max_evals
  ret = np.zeros((num_coordinates,))
  for point_j, group_j in zip(group_points, groupings):
    ret[group_j] = point_j
  return ret

def asy_add_ucb(gp, _, anc_data):
  """ Asynchronous Add UCB. """
  return _add_ucb(gp, gp.kernel, None, anc_data)

def syn_add_ucb(num_workers, gp, acq_optimiser, anc_data):
  """ Synchronous Add UCB. """
  # pylint: disable=unused-argument
  raise NotImplementedError('Not implemented Synchronous Add UCB yet.')

# UCB ------------------------------------------------------------------------------
def _get_gp_ucb_dim(gp):
  """ Returns the dimensionality of the dimension. """
  if hasattr(gp, 'ucb_dim') and gp.ucb_dim is not None:
    return gp.ucb_dim
  elif hasattr(gp.kernel, 'dim'):
    return gp.kernel.dim
  else:
    return 3.0

def _get_ucb_beta_th(dim, time_step):
  """ Computes the beta t for UCB based methods. """
  return np.sqrt(0.5 * dim * np.log(2 * dim * time_step + 1))

def asy_ucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB in the asyuential setting. """
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  def _ucb_acq(x):
    """ Computes the GP-UCB acquisition. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu + beta_th * sigma
  return _optimise_acquisition(_ucb_acq, acq_optimiser, anc_data)

def _halluc_ucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB using hallucinated inputs in the asynchronous
      setting. """
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  halluc_pts = _get_halluc_points(gp, anc_data.evals_in_progress)
  def _ucb_halluc_acq(x):
    """ Computes GP-UCB acquisition with hallucinated observations. """
    mu, sigma = gp.eval_with_hallucinated_observations(x, halluc_pts, uncert_form='std')
    return mu + beta_th * sigma
  return _optimise_acquisition(_ucb_halluc_acq, acq_optimiser, anc_data)

def asy_hucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB using hallucinated inputs in the asynchronous
      setting. """
  return _halluc_ucb(gp, acq_optimiser, anc_data)

def syn_hucb(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via Batch UCB in the synchronous setting. """
  recommendations = [asy_ucb(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    anc_data.evals_in_progress = recommendations
    recommendations.append(_halluc_ucb(gp, acq_optimiser, anc_data))
  return recommendations

def syn_ucbpe(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB-PE in the synchronous setting. """
  # Define some internal functions.
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  # 1. An LCB for the function
  def _ucbpe_lcb(x):
    """ An LCB for GP-UCB-PE. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu - beta_th * sigma
  # 2. A modified UCB for the function using hallucinated observations
  def _ucbpe_2ucb(x):
    """ A UCB for GP-UCB-PE. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu + 2 * beta_th * sigma
  # 3. UCB-PE acquisition for the 2nd point in the batch and so on.
  def _ucbpe_acq(x, yt_dot, halluc_pts):
    """ Acquisition for GP-UCB-PE. """
    _, halluc_stds = gp.eval_with_hallucinated_observations(x, halluc_pts,
                                                            uncert_form='std')
    return (_ucbpe_2ucb(x) > yt_dot).astype(np.double) * halluc_stds

  # Now the algorithm
  yt_dot_arg = _optimise_acquisition(_ucbpe_lcb, acq_optimiser, anc_data)
  yt_dot = _ucbpe_lcb(yt_dot_arg.reshape((-1, _get_gp_ucb_dim(gp))))
  recommendations = [asy_ucb(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    curr_acq = lambda x: _ucbpe_acq(x, yt_dot, np.array(recommendations))
    new_rec = _optimise_acquisition(curr_acq, acq_optimiser, anc_data)
    recommendations.append(new_rec)
  return recommendations

# EI stuff ----------------------------------------------------------------------------
def asy_ei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation based on GP-EI. """
  curr_best = anc_data.curr_max_val
  def _ei_acq(x):
    """ Acquisition for GP EI. """
    mu, sigma = gp.eval(x, uncert_form='std')
    Z = (mu - curr_best) / sigma
    return (mu - curr_best)*normal_distro.cdf(Z) + sigma*normal_distro.pdf(Z)
  return _optimise_acquisition(_ei_acq, acq_optimiser, anc_data)

def _halluc_ei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation based on GP-HEI using hallucinated points. """
  halluc_pts = _get_halluc_points(gp, anc_data.evals_in_progress)
  curr_best = anc_data.curr_max_val
  def _ei_halluc_acq(x):
    """ Computes the hallucinated EI acquisition. """
    mu, sigma = gp.eval_with_hallucinated_observations(x, halluc_pts, uncert_form='std')
    Z = (mu - curr_best) / sigma
    return (mu - curr_best)*normal_distro.cdf(Z) + sigma*normal_distro.pdf(Z)
  return _optimise_acquisition(_ei_halluc_acq, acq_optimiser, anc_data)

def asy_hei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via EI using hallucinated inputs in the asynchronous
      setting. """
  return _halluc_ei(gp, acq_optimiser, anc_data)

def syn_hei(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via EI in the synchronous setting. """
  recommendations = [asy_ei(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    anc_data.evals_in_progress = recommendations
    recommendations.append(_halluc_ei(gp, acq_optimiser, anc_data))
  return recommendations

# Random --------------------------------------------------------------------------------
def asy_rand(_, acq_optimiser, anc_data):
  """ Returns random values for the acquisition. """
  def _rand_eval(_):
    """ Acquisition for asy_rand. """
    return np.random.random((1,))
  return _optimise_acquisition(_rand_eval, acq_optimiser, anc_data)

def syn_rand(num_workers, gp, acq_optimiser, anc_data):
  """ Returns random values for the acquisition. """
  return [asy_rand(gp, acq_optimiser, anc_data) for _ in range(num_workers)]


# Multi-fidelity Strategies ==============================================================
def _get_fidel_to_opt_gp(mfgp, fidel_to_opt):
  """ Returns a GP for Boca that can passed to acq_optimise to optimise acquisition
      at fidel_to_opt. """
  boca_gp = Namespace()
  boca_gp.eval = lambda x, *args, **kwargs: mfgp.eval_at_fidel([fidel_to_opt] * len(x),
                                                               x, *args, **kwargs)
  boca_gp.kernel = mfgp.get_domain_kernel()
  return boca_gp

def _add_ucb_for_boca(mfgp, fidel_to_opt, mean_funcs, anc_data):
  """ Add UCB for BOCA. """
  # pylint: disable=undefined-loop-variable
  # TODO: this is repeating a lot of code from add_ucb. Fix this!
  # prelims
  domain_kernel_list = mfgp.domain_kernel.kernel_list
  groupings = mfgp.domain_kernel.groupings
  total_max_evals = anc_data.max_evals
  kern_scale = mfgp.kernel.hyperparams['scale']
  domain_bounds = anc_data.domain_bounds
  X_train = np.array(mfgp.XX)
  num_groups = len(domain_kernel_list)
  if mean_funcs is None:
    mean_funcs = lambda x: np.array([0] * len(x))
  if not hasattr(mean_funcs, '__iter__'):
    mean_funcs = [mean_funcs] * num_groups
  group_points = []
  num_coordinates = 0
  anc_data.max_evals = total_max_evals//num_groups
  K_fidel_Z_tr_to_f2o = mfgp.fidel_kernel(mfgp.ZZ, [fidel_to_opt])
  K_fidel_f2o_to_f2o = float(mfgp.fidel_kernel([fidel_to_opt], [fidel_to_opt]))

  # go through each element
  for group_j, kernel_j, mean_func_j in \
    zip(groupings, domain_kernel_list, mean_funcs):
    # Using python internal function here
    def _mf_add_ucb_acq_j(X_test_j):
      """ Acquisition for the j'th group. """
      betath_j = _get_add_ucb_beta_th(len(group_j), anc_data.t)
      X_train_j = X_train[:, group_j]
      K_tetr_domain_j = kernel_j(X_test_j, X_train_j)
      K_tetr_fidel_j = np.repeat(K_fidel_Z_tr_to_f2o.T, len(X_test_j), axis=0)
      K_tetr_j = kern_scale * K_tetr_fidel_j * K_tetr_domain_j
      pred_mean_j = K_tetr_j.dot(mfgp.alpha) + mean_func_j(X_test_j)
      K_tete_j = kern_scale * K_fidel_f2o_to_f2o * kernel_j(X_test_j, X_test_j)
      V_j = solve_lower_triangular(mfgp.L, K_tetr_j.T)
      post_covar_j = K_tete_j - V_j.T.dot(V_j)
      post_covar_j = get_post_covar_from_raw_covar(post_covar_j, mfgp.noise_var,
                                                   mfgp.kernel.is_guaranteed_psd())
      post_std_j = np.sqrt(np.diag(post_covar_j))
      return pred_mean_j + betath_j * post_std_j
    # Objatin the jth group
    acq_optimiser_j = _get_add_acq_optimiser(domain_bounds[group_j],
                                             anc_data.acq_opt_method)
    point_j = _optimise_acquisition(_mf_add_ucb_acq_j, acq_optimiser_j, anc_data)
    group_points.append(point_j)
    num_coordinates += len(point_j)

  # return
  anc_data.max_evals = total_max_evals
  ret = np.zeros((num_coordinates,))
  for point_j, group_j in zip(group_points, groupings):
    ret[group_j] = point_j
  return ret

def asy_add_ucb_for_boca(mfgp, fidel_to_opt, anc_data):
  """ Asynchronous Add UCB. """
  return _add_ucb_for_boca(mfgp, fidel_to_opt, None, anc_data)

def syn_add_ucb_for_boca(mfgp, fidel_to_opt, anc_data):
  """ Synchronous Add UCB. """
  # pylint: disable=unused-argument
  raise NotImplementedError('Not implemented Synchronous Add UCB yet.')

def boca(select_pt_func, acq_optimise, mfgp, anc_data, func_caller):
  """ Uses the BOCA strategy to pick the next point and fidelity as described in
      Kandasamy et al. 2017 "Multi-fidelity Bayesian Optimisation with Continuous
      Approximations (https://arxiv.org/pdf/1703.06240.pdf).
      We have some additional heuristics implemented as described in the appendix.
  """
  if anc_data.acq == 'add_ucb':
    next_eval_point = asy_add_ucb_for_boca(mfgp, func_caller.fidel_to_opt, anc_data)
  else:
    fidel_to_opt_gp = _get_fidel_to_opt_gp(mfgp, func_caller.fidel_to_opt)
    next_eval_point = select_pt_func(fidel_to_opt_gp, acq_optimise, anc_data)
  candidate_fidels, cost_ratios = func_caller.get_candidate_fidels_and_cost_ratios(
                                    next_eval_point, filter_by_cost=True)
  num_candidates = len(candidate_fidels)
  cost_ratios = np.array(cost_ratios)
  sqrt_cost_ratios = np.sqrt(cost_ratios)
  information_gaps = np.array(func_caller.get_information_gap(candidate_fidels))
  _, cand_fidel_stds = mfgp.eval_at_fidel(candidate_fidels,
                                          [next_eval_point] * num_candidates,
                                          uncert_form='std')
  cand_fidel_stds = cand_fidel_stds / np.sqrt(mfgp.kernel.hyperparams['scale'])
  std_thresholds = anc_data.boca_thresh_coeff * anc_data.y_range * \
                   sqrt_cost_ratios * information_gaps
  high_std_idxs = cand_fidel_stds > std_thresholds
  qualifying_idxs = np.where(high_std_idxs)[0]
  if len(qualifying_idxs) == 0:
    next_eval_fidel = func_caller.fidel_to_opt
  else:
    # If the cost_ratio is larger than a threshold, then just evaluate at fidel to opt.
    qualifying_fidels = [candidate_fidels[idx] for idx in qualifying_idxs]
    # boca 0
    qualifying_sqrt_cost_ratios = sqrt_cost_ratios[qualifying_idxs]
    qualifying_cost_ratios = cost_ratios[qualifying_idxs]
    next_eval_fidel_idx = qualifying_sqrt_cost_ratios.argmin()
    # Select the fidel
    if qualifying_cost_ratios[next_eval_fidel_idx] > \
       anc_data.boca_max_low_fidel_cost_ratio:
      next_eval_fidel = func_caller.fidel_to_opt
    else:
      next_eval_fidel = qualifying_fidels[next_eval_fidel_idx]
  return next_eval_fidel, next_eval_point


# Put all of them into the following namespaces.
syn = Namespace(
  # UCB
  hucb=syn_hucb,
  ucbpe=syn_ucbpe,
  ucb=syn_hucb,
  add_ucb=syn_add_ucb,
  # TS
  ts=syn_ts,
  # EI
  hei=syn_hei,
  # Rand
  rand=syn_rand,
  )

asy = Namespace(
  # UCB
  ucb=asy_ucb,
  hucb=asy_hucb,
  add_ucb=asy_add_ucb,
  add_hucb=asy_add_ucb,
  # EI
  ei=asy_ei,
  hei=asy_hei,
  # TS
  ts=asy_ts,
  hts=asy_hts,
  # Rand
  rand=asy_rand,
  )

seq = Namespace(
  ucb=asy_ucb,
  ts=asy_ts,
  add_ucb=asy_ucb,
  )

