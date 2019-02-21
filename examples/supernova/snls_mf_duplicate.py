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
from demos_real.supernova.snls_mf import snls_avg_log_likelihood


NUM_DATA_MAX = 192
RESOLUTION_MAX = 1000


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

