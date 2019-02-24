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

from snls_mf import snls_avg_log_likelihood

def objective(x):
  """ Multi-fidelity objective. """
  return snls_avg_log_likelihood(x)

