"""
  Computes the likelihood of three cosmological parameters (Hubble constant, dark matter
  fraction, and dark energy fraction on the Davis et al. 2007 dataset (see below) and the
  Robertson-Walker metric.
  -- kandasamy@cs.cmu.edu

  This code is released under the MIT license (https://en.wikipedia.org/wiki/MIT_License).

  If you use any part of this implementation, please cite the following papers.
  - Kandasamy K, Dasarathy G, Schneider J, Poczos B. "Multi-fidelity Bayesian Optimisation
    with Continuous Approximations", International Conference on Machine Learning 2017
  - (dataset) Davis TM et al. "Scrutinizing Exotic Cosmological Models Using ESSENCE
    Supernova Data Combined with Other Cosmological Probes", Astrophysical Journal 2007
  - (likelihood approximation 1) Robertson H. P. "An Intepretation of Page's ``New
    Relativity``", 1936.
  - (likelihood approximation 2) Shchigolev V K. "Calculating Luminosity Distance versus
    Redshift in FLRW Cosmology via Homotopy Perturbation Method", 2016.
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

import numpy as np
from scipy.stats import norm
from scipy import integrate

# constants
SPEED_OF_LIGHT = 299792.48
INTEGRATE_METHOD = integrate.trapz
RESOLUTION_MAX = 1e5
NUM_DATA_MAX = 192
DFLT_APPROXIMATION_METHOD = 'shchigolev'
# DFLT_APPROXIMATION_METHOD = 'robertson'

# Load data
try:
  import os
  curr_dir_path = os.path.dirname(os.path.realpath(__file__))
  data_path = os.path.join(curr_dir_path, 'davisdata')
  OBS_DATA = np.loadtxt(data_path)
except IOError:
  print(('Could not load file %s. Make sure the file davisdata is in the same ' +
         'directory as this file or pass the dataset to the function.')%(data_path))


def snls_avg_log_likelihood(eval_pt, resolution=None, num_obs_to_use=None, obs_data=None,
                            approximation_method=DFLT_APPROXIMATION_METHOD,
                            to_shuffle_data_when_using_a_subset=True):
  """ Returns the *average* log likelihood for the super nova dataset. To obtain the
      log likelihood multiply by the length of obs_data.
        - eval_pt: is an array of length 3 with values for the Hubble constant, dark
                   matter fraction, and dark energy fraction in that order.
        - resolution: the resolution at which to perform the integration.
        - num_obs_to_use: the number of observations in obs_data to use.
        - obs_data: the observations. (See accompanying file davisdata)
  """
  # pylint: disable=too-many-locals
  # Prelims -------------------------------
  if obs_data is None:
    obs_data = OBS_DATA
  num_obs = obs_data.shape[0]
  if resolution is None:
    resolution = RESOLUTION_MAX
  if num_obs_to_use is None:
    num_obs_to_use = num_obs
  # Make sure they are both integers
  resolution = int(resolution)
  num_obs_to_use = int(num_obs_to_use)
  if num_obs_to_use < num_obs and to_shuffle_data_when_using_a_subset:
    obs_data = np.copy(obs_data)
    np.random.shuffle(obs_data)
  # Decompose Data
  obs_data_to_use = obs_data[0:num_obs_to_use, :]
  red_shifts = obs_data_to_use[:, 0]
  obs_locs = obs_data_to_use[:, 1]
  obs_errors = obs_data_to_use[:, 2]
  # Separate out the eval_pt
  hubble_const = eval_pt[0]               # Hubble constant
  omega_m = eval_pt[1]                    # Dark matter fraction
  omega_l = eval_pt[2]                    # Dark energy fraction
  omega_r = max(0, 1 - omega_m - omega_l) # radiation fraction
  curr_lum_means = [0] * num_obs_to_use
  # Define the function to be integrated
  if approximation_method == 'robertson':
    f = lambda t: 1/ np.sqrt(omega_m * (1+t)**3 + omega_l)
  elif approximation_method == 'shchigolev':
    f = lambda t: 1/ np.sqrt(omega_r * (1+t)**4 + omega_m * (1+t)**3 + omega_l)
  else:
    raise ValueError('Unknown approximation method %s.'%(approximation_method))
  # Iterate through each point in the dataset.
  for obs_idx in range(num_obs_to_use):
    dlc = SPEED_OF_LIGHT * (1 + red_shifts[obs_idx])/hubble_const
    integrate_grid = np.linspace(0, red_shifts[obs_idx], resolution)
    integrate_grid_res = integrate_grid[1] - integrate_grid[0]
    integrate_values = [f(pt) for pt in integrate_grid]
    dli = INTEGRATE_METHOD(integrate_values, dx=integrate_grid_res)
    dl = dlc * dli
    curr_lum_means[obs_idx] = 5 * np.log10(dl) + 25
  obs_likelihoods = norm.pdf(obs_locs, curr_lum_means, obs_errors)
  obs_log_likelihoods = np.log(obs_likelihoods)
  # Clip at -1000
  obs_log_likelihoods = np.clip(obs_log_likelihoods, -1000, np.inf)
  avg_log_likelihood = obs_log_likelihoods.mean()
  return avg_log_likelihood


def _get_resoltion_from_z0(z0):
  """ Return resolution. """
  return 10**z0


# Cost for multi-fidelity
def cost(z):
  """ Cost function for multi-fidelity. The cost is linear in both the resolution and
      the amount of data used. The 1e-5 is to ensure that the cost function is never 0.
  """
  resolution = _get_resoltion_from_z0(z[0])
  return 1e-5 + (resolution/float(RESOLUTION_MAX)) * (z[1]/float(NUM_DATA_MAX))


def objective(z, x):
  """ Multi-fidelity objective.
      z[0] is assumed to be the log10 resolution and z[1] the amount of data to be used.
  """
  resolution = _get_resoltion_from_z0(z[0])
  return snls_avg_log_likelihood(x, resolution, z[1])

