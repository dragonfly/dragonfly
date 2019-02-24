"""
  Acquisition functions for Bayesian Optimisation.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name

from argparse import Namespace
from copy import copy
import numpy as np
from scipy.stats import norm as normal_distro
# Local
from ..utils.general_utils import solve_lower_triangular
from ..gp.gp_core import get_post_covar_from_raw_covar
from ..exd.domains import EuclideanDomain
from ..exd.exd_utils import maximise_with_method

# TODO: implement using different samples for synchronous methods


# Some utilities we will use for all acquisitions below ---------------------------------
def maximise_acquisition(acq_fn, anc_data, *args, **kwargs):
  """ Maximises the acquisition and returns the highest point. acq_fn is the acquisition
      function to be maximised. anc_data is a namespace which contains ancillary data.
  """
  # pylint: disable=unbalanced-tuple-unpacking
  acq_opt_method = anc_data.acq_opt_method
  if anc_data.domain.get_type() == 'euclidean':
    if acq_opt_method in ['rand']:
      acquisition = acq_fn
    else:
      # these methods cannot handle vectorised functions.
      acquisition = lambda x: acq_fn(x.reshape((1, -1)))
  elif anc_data.domain.get_type() == 'cartesian_product':
    # these methods cannot handle vectorised functions.
    acquisition = lambda x: acq_fn([x])
  _, opt_pt = maximise_with_method(acq_opt_method, acquisition, anc_data.domain,
                                   anc_data.max_evals, *args, **kwargs)
  return opt_pt


def _get_gp_eval_for_parallel_strategy(gp, anc_data, uncert_form='std'):
  """ Returns the evaluation function of the gp depending on the parallel strategy and
      the evaluations in progress.
  """
  # 1. With hallucinations
  def _get_halluc_gp_eval(_gp, _halluc_pts, _uncert_form):
    """ Hallucinated GP eval. """
    return lambda x: _gp.eval_with_hallucinated_observations(x, _halluc_pts,
                                                             uncert_form=_uncert_form)
  # Ordinary eval of gp
  def _get_naive_gp_eval(_gp, _uncert_form):
    """ Naive GP eval. """
    return lambda x: _gp.eval(x, uncert_form=_uncert_form)
  # Check parallelisation strategy and return
  if anc_data.handle_parallel == 'halluc' and \
    len(anc_data.eval_points_in_progress) > 0:
    if anc_data.is_mf:
      return _get_halluc_gp_eval(gp, anc_data.eval_fidel_points_in_progress, uncert_form)
    else:
      return _get_halluc_gp_eval(gp, anc_data.eval_points_in_progress, uncert_form)
  else:
    return _get_naive_gp_eval(gp, uncert_form)


def get_gp_sampler_for_parallel_strategy(gp, anc_data):
  """ Returns a function that can draw samples from the posterior gp depending on the
      parallel strategy and the evaluations in progress.
  """
  # 1. With hallucinations
  def _get_halluc_gp_draw_samples(_gp, _halluc_pts):
    """ Hallucinated sampler. """
    return lambda x: _gp.draw_samples_with_hallucinated_observations(1, x,
                                                                     _halluc_pts).ravel()
  def _get_naive_gp_draw_samples(_gp):
    """ Naive sampler. """
    return lambda x: _gp.draw_samples(1, x).ravel()
  # Check parallelisation strategy and return
  if anc_data.handle_parallel == 'halluc' and \
    len(anc_data.eval_points_in_progress) > 0:
    if anc_data.is_mf:
      return _get_halluc_gp_draw_samples(gp, anc_data.eval_fidel_points_in_progress)
    else:
      return _get_halluc_gp_draw_samples(gp, anc_data.eval_points_in_progress)
  else:
    return _get_naive_gp_draw_samples(gp)


def _get_syn_recommendations_from_asy(asy_acq, num_workers, list_of_gps, anc_datas):
  """ Returns a batch of (synchronous recommendations from an asynchronous acquisition.
  """
  def _get_next_and_append(_list_of_objects):
    """ Internal function to return current gp and list of gps. """
    ret = _list_of_objects.pop(0)
    _list_of_objects = _list_of_objects + [ret]
    return ret, _list_of_objects
  # If list_of_gps is not a list, then make it a list.
  if not hasattr(list_of_gps, '__iter__'):
    list_of_gps = [list_of_gps] * num_workers
  if not hasattr(anc_datas, '__iter__'):
    anc_datas = [anc_datas] * num_workers
  # Create copies
  list_of_gps = [copy(gp) for gp in list_of_gps]
  anc_datas = [copy(ad) for ad in anc_datas]
  # Get first recommendation
  next_gp, list_of_gps = _get_next_and_append(list_of_gps)
  next_anc_data, anc_datas = _get_next_and_append(anc_datas)
  recommendations = [asy_acq(next_gp, next_anc_data)]
  for _ in range(1, num_workers):
    next_gp, list_of_gps = _get_next_and_append(list_of_gps)
    next_anc_data, anc_datas = _get_next_and_append(anc_datas)
    next_anc_data.eval_points_in_progress = recommendations
    recommendations.append(asy_acq(next_gp, next_anc_data))
  return recommendations


# Thompson sampling ---------------------------------------------------------------
def asy_ts(gp, anc_data):
  """ Returns a recommendation via TS in the asyuential setting. """
  anc_data = copy(anc_data)
  # Always use a random optimiser with a vectorised sampler for TS.
  if anc_data.acq_opt_method != 'rand':
    anc_data.acq_opt_method = 'rand'
    anc_data.max_evals = 4 * anc_data.max_evals
  gp_sample = get_gp_sampler_for_parallel_strategy(gp, anc_data)
  return maximise_acquisition(gp_sample, anc_data, vectorised=True)

def syn_ts(num_workers, list_of_gps, anc_datas):
  """ Returns a batch of recommendations via TS in the synchronous setting. """
  return _get_syn_recommendations_from_asy(asy_ts, num_workers, list_of_gps, anc_datas)


# Add-UCB --------------------------------------------------------------------------
def _get_add_ucb_beta_th(dim, time_step):
  """ Computes the beta t for UCB based methods. """
  return np.sqrt(0.2 * dim * np.log(2 * dim * time_step + 1))

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
    anc_data_j = copy(anc_data)
    anc_data_j.domain = EuclideanDomain(domain_bounds[group_j])
    point_j = maximise_acquisition(_add_ucb_acq_j, anc_data_j)
    group_points.append(point_j)
    num_coordinates += len(point_j)

  # Now return
  anc_data.max_evals = total_max_evals
  ret = np.zeros((num_coordinates,))
  for point_j, group_j in zip(group_points, groupings):
    ret[group_j] = point_j
  return ret

def asy_add_ucb(gp, anc_data):
  """ Asynchronous Add UCB. """
  return _add_ucb(gp, gp.kernel, None, anc_data)

def syn_add_ucb(num_workers, list_of_gps, anc_datas):
  """ Synchronous Add UCB. """
  # pylint: disable=unused-argument
  return _get_syn_recommendations_from_asy(asy_add_ucb, num_workers, list_of_gps,
                                           anc_datas)

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

def asy_ucb(gp, anc_data):
  """ Returns a recommendation via UCB in the asyuential setting. """
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  gp_eval = _get_gp_eval_for_parallel_strategy(gp, anc_data, 'std')
  def _ucb_acq(x):
    """ Computes the GP-UCB acquisition. """
    mu, sigma = gp_eval(x)
    return mu + beta_th * sigma
  return maximise_acquisition(_ucb_acq, anc_data)

def syn_ucb(num_workers, list_of_gps, anc_datas):
  """ Returns a recommendation via Batch UCB in the synchronous setting. """
  return _get_syn_recommendations_from_asy(asy_ucb, num_workers, list_of_gps, anc_datas)

# PI  ----------------------------------------------------------------------------
def asy_pi(gp, anc_data):
  """ Returns a recommendation based on PI (probability of improvement). """
  curr_best = anc_data.curr_max_val
  gp_eval = _get_gp_eval_for_parallel_strategy(gp, anc_data, 'std')
  # PI acquisition with hallucinated observations
  def _pi_acq(x):
    """ Acquisition for GP EI. """
    mu, sigma = gp_eval(x)
    return normal_distro.cdf((mu - curr_best) / sigma)
  return maximise_acquisition(_pi_acq, anc_data)

def syn_pi(num_workers, list_of_gps, anc_datas):
  """ Returns a recommendation via EI in the synchronous setting. """
  return _get_syn_recommendations_from_asy(asy_pi, num_workers, list_of_gps, anc_datas)


# EI  ----------------------------------------------------------------------------
def _expected_improvement_for_norm_diff(norm_diff):
  """ The expected improvement. """
  return norm_diff * normal_distro.cdf(norm_diff) + normal_distro.pdf(norm_diff)

def asy_ei(gp, anc_data):
  """ Returns a recommendation based on GP-EI. """
  curr_best = anc_data.curr_max_val
  gp_eval = _get_gp_eval_for_parallel_strategy(gp, anc_data, 'std')
  # EI acquisition with hallucinated observations
  def _ei_acq(x):
    """ Acquisition for GP EI. """
    mu, sigma = gp_eval(x)
    norm_diff = (mu - curr_best) / sigma
    return sigma * _expected_improvement_for_norm_diff(norm_diff)
  return maximise_acquisition(_ei_acq, anc_data)

def syn_ei(num_workers, list_of_gps, anc_datas):
  """ Returns a recommendation via EI in the synchronous setting. """
  return _get_syn_recommendations_from_asy(asy_ei, num_workers, list_of_gps, anc_datas)


# TTEI ----------------------------------------------------------------------------------
def _ttei(gp_eval, anc_data, ref_point):
  """ Computes the arm that is expected to do best over ref_point. """
  ref_mean, ref_std = gp_eval([ref_point])
  ref_mean = float(ref_mean)
  ref_std = float(ref_std)
  def _tt_ei_acq(x):
    """ Acquisition for TTEI. """
    mu, sigma = gp_eval(x)
    comb_std = np.sqrt(ref_std**2 + sigma**2)
    norm_diff = (mu - ref_mean)/comb_std
    return comb_std * _expected_improvement_for_norm_diff(norm_diff)
  return maximise_acquisition(_tt_ei_acq, anc_data)

def asy_ttei(gp, anc_data):
  """ Top-Two expected improvement. """
  if np.random.random() < 0.5:
    # With probability 1/2, return the EI point
    return asy_ei(gp, anc_data)
  else:
    max_acq_opt_evals = anc_data.max_evals
    anc_data = copy(anc_data)
    anc_data.max_evals = max_acq_opt_evals//2
    ei_argmax = asy_ei(gp, anc_data)
    # Now return the second argmax
    gp_eval = _get_gp_eval_for_parallel_strategy(gp, anc_data, 'std')
    return _ttei(gp_eval, anc_data, ei_argmax)

def syn_ttei(num_workers, list_of_gps, anc_data):
  """ Returns a recommendation via TTEI in the synchronous setting. """
  return _get_syn_recommendations_from_asy(asy_ttei, num_workers, list_of_gps, anc_data)

# Random --------------------------------------------------------------------------------
def asy_rand(_, anc_data):
  """ Returns random values for the acquisition. """
  def _rand_eval(_):
    """ Acquisition for asy_rand. """
    return np.random.random((1,))
  return maximise_acquisition(_rand_eval, anc_data)

def syn_rand(num_workers, list_of_gps, anc_data):
  """ Returns random values for the acquisition. """
  return _get_syn_recommendations_from_asy(asy_rand, num_workers, list_of_gps, anc_data)


# Multi-fidelity Strategies ==============================================================
def _get_fidel_to_opt_gp(mfgp, fidel_to_opt):
  """ Returns a GP for Boca that can be used to optimise the acquisition
      at fidel_to_opt. """
  boca_gp = Namespace()
  boca_gp.eval = lambda x, *args, **kwargs: mfgp.eval_at_fidel([fidel_to_opt] * len(x),
                                                               x, *args, **kwargs)
  boca_gp.eval_with_hallucinated_observations = \
    lambda x, halluc_fidel_pts, *args, **kwargs: mfgp.eval_with_hallucinated_observations(
      mfgp.get_ZX_from_ZZ_XX([fidel_to_opt] * len(x), x), halluc_fidel_pts,
      *args, **kwargs)
  boca_gp.draw_samples = lambda n, x, *args, **kwargs: mfgp.draw_samples(n,
                      mfgp.get_ZX_from_ZZ_XX([fidel_to_opt] * len(x), x), *args, **kwargs)
  boca_gp.draw_samples_with_hallucinated_observations = \
                lambda n, x, halluc_fidel_pts, *args, **kwargs: \
                  mfgp.draw_samples_with_hallucinated_observations(n,
                    mfgp.get_ZX_from_ZZ_XX([fidel_to_opt] * len(x), x), halluc_fidel_pts,
                                           *args, **kwargs)
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
    anc_data_j = copy(anc_data)
    anc_data_j.domain = EuclideanDomain(domain_bounds[group_j])
    point_j = maximise_acquisition(_mf_add_ucb_acq_j, anc_data_j)

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

def syn_add_ucb_for_boca(num_workers, list_of_mfgps, fidel_to_opt, anc_data):
  """ Synchronous Add UCB. """
  # pylint: disable=unused-argument
  raise NotImplementedError('Not Implemented Yet!')

def boca(select_pt_func, mfgp, anc_data, func_caller):
  """ Uses the BOCA strategy to pick the next point and fidelity as described in
      Kandasamy et al. 2017 "Multi-fidelity Bayesian Optimisation with Continuous
      Approximations (https://arxiv.org/pdf/1703.06240.pdf).
      We have some additional heuristics implemented as described in the appendix.
  """
  if anc_data.curr_acq == 'add_ucb':
    next_eval_point = asy_add_ucb_for_boca(mfgp, func_caller.fidel_to_opt, anc_data)
  else:
    fidel_to_opt_gp = _get_fidel_to_opt_gp(mfgp, func_caller.fidel_to_opt)
    next_eval_point = select_pt_func(fidel_to_opt_gp, anc_data)
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
  ucb=syn_ucb,
  add_ucb=syn_add_ucb,
  ei=syn_ei,
  pi=syn_pi,
  ttei=syn_ttei,
  ts=syn_ts,
  rand=syn_rand,
  )

asy = Namespace(
  ucb=asy_ucb,
  add_ucb=asy_add_ucb,
  ei=asy_ei,
  pi=asy_pi,
  ttei=asy_ttei,
  ts=asy_ts,
  rand=asy_rand,
  )

seq = Namespace(
  ucb=asy_ucb,
  add_ucb=asy_add_ucb,
  ei=asy_ei,
  pi=asy_pi,
  ttei=asy_ttei,
  ts=asy_ts,
  rand=asy_rand,
  )

