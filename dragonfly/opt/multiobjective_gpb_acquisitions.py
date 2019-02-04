"""
  Multi Objective Acquisition functions for Bayesian Optimisation.
  - bparia@cs.cmu.edu
"""

from __future__ import division

# pylint: disable=invalid-name

from argparse import Namespace
from copy import copy
import numpy as np
# Local
from .gpb_acquisitions import maximise_acquisition
from .gpb_acquisitions import get_gp_sampler_for_parallel_strategy


# Multi-Objective Thompson Sampling ========================================
def mo_lin_asy_ts(gps, anc_data):
  """ Returns a recommendation via TS with Linear Scalarization in the
  asynchronous setting for MOO """
  anc_data = copy(anc_data)
  # Always use a random optimiser with a vectorised sampler for TS.
  if anc_data.acq_opt_method != 'rand':
    anc_data.acq_opt_method = 'rand'
    anc_data.max_evals = 4 * anc_data.max_evals
  gp_samples = []
  for gp in gps:
    gp_sample = get_gp_sampler_for_parallel_strategy(gp, anc_data)
    gp_samples.append(gp_sample)

  def acquisition(x):
    """ Computes the scalar acquisition function for the given weight
        vector and the reference point.
    """
    s = 0.0
    for gp_sample, weight in zip(gp_samples, anc_data.obj_weights):
      s += gp_sample(x) * weight
    return s

  return maximise_acquisition(acquisition, anc_data, vectorised=True)


def mo_tch_asy_ts(gps, anc_data):
  """ Returns a recommendation via TS with Tchebychev Scalarization in the
  asynchronous setting for MOO """
  anc_data = copy(anc_data)
  # Always use a random optimiser with a vectorised sampler for TS.
  if anc_data.acq_opt_method != 'rand':
    anc_data.acq_opt_method = 'rand'
    anc_data.max_evals = 4 * anc_data.max_evals
  gp_samples = []
  for gp in gps:
    gp_sample = get_gp_sampler_for_parallel_strategy(gp, anc_data)
    gp_samples.append(gp_sample)
  # TS acquisition
  def acquisition(x):
    """ Computes the scalar acquisition function for the given weight
        vector and the reference point.
    """
    s = np.full((len(x), ), np.inf)
    for gp_sample, weight, ref in \
        zip(gp_samples, anc_data.obj_weights, anc_data.reference_point):
      s = np.minimum(s, (gp_sample(x) - ref) / weight)
    return s
  # return
  return maximise_acquisition(acquisition, anc_data, vectorised=True)


# Multi-Objective UCB ========================================
def _get_ucb_beta_th(dim, time_step):
  """ Computes the beta t for UCB based methods. """
  return np.sqrt(0.2 * dim * np.log(2 * dim * time_step + 1))


def mo_lin_asy_ucb(gps, anc_data):
  """ Returns a recommendation via UCB with Linear Scalarization in the
  asynchronous setting for MOO """
  beta_th = _get_ucb_beta_th(anc_data.domain.dim, anc_data.t)
  def acquisition(x):
    """ Computes the GP-UCB acquisition. """
    mu_tot = 0.0
    sigma2_tot = 0.0
    for gp, weight in zip(gps, anc_data.obj_weights):
      mu, sigma = gp.eval(x, uncert_form='std')
      mu_tot += mu * weight
      sigma2_tot += sigma * sigma * weight**2
    ret = mu_tot + beta_th * np.sqrt(sigma2_tot)
    return ret
  return maximise_acquisition(acquisition, anc_data)


def mo_tch_asy_ucb(gps, anc_data):
  """ Returns a recommendation via UCB with Linear Scalarization in the
  asynchronous setting for MOO """
  beta_th = _get_ucb_beta_th(anc_data.domain.dim, anc_data.t)
  def acquisition(x):
    """ Computes the GP-UCB acquisition. """
    ret = np.asarray([np.inf for _ in range(len(x))])
    for gp, weight, ref in \
        zip(gps, anc_data.obj_weights, anc_data.reference_point):
      mu, sigma2 = gp.eval(x, uncert_form='std')
      ucb = mu + beta_th * np.sqrt(sigma2) - ref
      ret = np.minimum(ret, ucb / weight)
    return ret
  return maximise_acquisition(acquisition, anc_data)


asy = Namespace(
  lin_ts=mo_lin_asy_ts,
  tch_ts=mo_tch_asy_ts,
  lin_ucb=mo_lin_asy_ucb,
  tch_ucb=mo_tch_asy_ucb,
)

# TODO: implement synchronous versions. Need to implement a version of
# _get_syn_recommendations_from_asy in gpb_acquisitions.
syn = Namespace()

seq = Namespace(
  lin_ts=mo_lin_asy_ts,
  tch_ts=mo_tch_asy_ts,
  lin_ucb=mo_lin_asy_ucb,
  tch_ucb=mo_tch_asy_ucb,
)

