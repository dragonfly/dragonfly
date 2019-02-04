"""
  Metropolis Hastings Sampler -- Adapted from pymc3 library.
  https://github.com/pymc-devs/pymc3
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=unexpected-keyword-arg
# pylint: disable=redefined-variable-type

import scipy.linalg
import numpy as np
import numpy.random as nr

__all__ = ['Metropolis', 'BinaryMetropolis', 'NormalProposal', 'CauchyProposal',
           'LaplaceProposal', 'PoissonProposal', 'MultivariateNormalProposal']

# Available proposal distributions for Metropolis

class Proposal(object):
  """ Base class for Proposal Distribution."""
  def __init__(self, s):
    self.s = s


class NormalProposal(Proposal):
  """ Normal Proposal Distribution."""
  def __call__(self):
    return nr.normal(scale=self.s)


class UniformProposal(Proposal):
  """ Uniform Proposal Distribution."""
  def __call__(self):
    return nr.uniform(low=-self.s, high=self.s, size=len(self.s))


class CauchyProposal(Proposal):
  """ Cauchy Proposal Distribution."""
  def __call__(self):
    return nr.standard_cauchy(size=np.size(self.s)) * self.s


class LaplaceProposal(Proposal):
  """ Laplace Proposal Distribution."""
  def __call__(self):
    size = np.size(self.s)
    return (nr.standard_exponential(size=size) - nr.standard_exponential(size=size))\
            *self.s


class PoissonProposal(Proposal):
  """ Poisson Proposal Distribution."""
  def __call__(self):
    return nr.poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
  """ Multivariate Normal Proposal Distribution."""
  def __init__(self, s):
    n, m = s.shape
    if n != m:
      raise ValueError("Covariance matrix is not symmetric.")
    self.n = n
    self.chol = scipy.linalg.cholesky(s, lower=True)
    super(MultivariateNormalProposal, self).__init__()

  def __call__(self, num_draws=None):
    if num_draws is not None:
      b = np.random.randn(self.n, num_draws)
      return np.dot(self.chol, b).T
    else:
      b = np.random.randn(self.n)
      return np.dot(self.chol, b)


class Metropolis(object):
  """
  Metropolis-Hastings sampling step

  Parameters
  ----------
  S : standard deviation or covariance matrix
    Some measure of variance to parameterize proposal distribution
  proposal_dist : function
    Function that returns zero-mean deviates when parameterized with
    S (and n). Defaults to normal.
  scaling : scalar or array
    Initial scale factor for proposal. Defaults to 1.
  tune : bool
    Flag for tuning. Defaults to True.
  tune_interval : int
    The frequency of tuning. Defaults to 100 iterations.
  """

  def __init__(self, model, discrete=False, S=None, proposal_dist=None, scaling=1.,\
               tune=True, tune_interval=100):

    if S is None:
      S = np.ones(model.get_dim())

    if proposal_dist is not None:
      self.proposal_dist = proposal_dist(S)
    elif S.ndim == 1:
      self.proposal_dist = NormalProposal(S)
    elif S.ndim == 2:
      self.proposal_dist = MultivariateNormalProposal(S)
    else:
      raise ValueError("Invalid rank for variance: %s" % S.ndim)

    self.model = model
    self.scaling = np.atleast_1d(scaling).astype('d')
    self.tune = tune
    self.tune_interval = tune_interval
    self.steps_until_tune = tune_interval
    self.accepted = 0
    self.discrete = discrete

    super(Metropolis, self).__init__()

  def _sample(self, q0):
    """
    Helper function which implements metropolis sampler.
    """
    if not self.steps_until_tune and self.tune:
      # Tune scaling parameter
      self.scaling = tune_params(self.scaling, self.accepted / float(self.tune_interval))
      # Reset counter
      self.steps_until_tune = self.tune_interval
      self.accepted = 0

    delta = self.proposal_dist() * self.scaling
    if self.discrete:
      delta = np.round(delta, 0).astype('int64')
      q0 = q0.astype('int64')
      q = (q0 + delta).astype('int64')
    else:
      q = q0 + delta

    accept = delta_logp(self.model, q, q0)
    q_new, accepted = metrop_select(accept, q, q0)
    self.accepted += accepted

    self.steps_until_tune -= 1

    return q_new

  def sample(self, q0, num_samples=1):
    """
    sample -- function which populates samples
    by calling function _sample iteratively
    """
    if num_samples is None:
      num_samples = 1
    if not hasattr(q0, '__len__'):
      q0 = np.array([q0])

    samples = np.zeros([num_samples, len(q0)])
    for i in range(num_samples):
      samples[i] = self._sample(q0)
      q0 = samples[i]

    return samples


class BinaryMetropolis(object):
  """Metropolis-Hastings optimized for binary variables

  Parameters
  ----------
  scaling : scalar or array
      Initial scale factor for proposal. Defaults to 1.
  """

  def __init__(self, model, scaling=1.):
    self.model = model
    self.scaling = scaling

    super(BinaryMetropolis, self).__init__()

  def _sample(self, q0):
    """
    Helper function which implements binary metropolis sampler.
    """
    # Convert adaptive_scale_factor to a jump probability
    p_jump = 1. - .5 ** self.scaling

    rand_array = nr.random(q0.shape)
    q = np.copy(q0)

    # Locations where switches occur, according to p_jump
    switch_locs = (rand_array < p_jump)
    q[switch_locs] = True - q[switch_locs]

    accept = self.model.logp(q) - self.model.logp(q0)
    q_new, _ = metrop_select(np.exp(accept), q, q0)

    return q_new

  def sample(self, q0, num_samples=1):
    """
    sample -- function which populates samples
    by calling function _sample iteratively
    """
    if num_samples is None:
      num_samples = 1
    if not hasattr(q0, '__len__'):
      q0 = np.array([q0])

    samples = np.zeros([num_samples, len(q0)])
    for i in range(num_samples):
      samples[i] = self._sample(q0)
      q0 = samples[i]

    return samples


def metrop_select(mr, q, q0):
  """Perform rejection/acceptance step for Metropolis class samplers.

  Returns the new sample q if a uniform random number is less than the
  metropolis acceptance rate (`mr`), and the old sample otherwise, along
  with a boolean indicating whether the sample was accepted.

  Parameters
  ----------
  mr : float, Metropolis acceptance rate
  q : proposed sample
  q0 : current sample

  Returns
  -------
  q or q0
  """
  # Compare acceptance ratio to uniform random number
  if np.isfinite(mr) and nr.uniform() < mr:
    return q, True
  else:
    return q0, False


def tune_params(scale, acc_rate):
  """
  Tunes the scaling parameter for the proposal distribution
  according to the acceptance rate over the last tune_interval:

  Rate  Variance adaptation
  ----  -------------------
  <0.001    x 0.1
  <0.05     x 0.5
  <0.2      x 0.9
  >0.5      x 1.1
  >0.75     x 2
  >0.95     x 10

  """

  # Switch statement
  if acc_rate < 0.001:
    # reduce by 90 percent
    scale *= 0.1
  elif acc_rate < 0.05:
    # reduce by 50 percent
    scale *= 0.5
  elif acc_rate < 0.2:
    # reduce by ten percent
    scale *= 0.9
  elif acc_rate > 0.95:
    # increase by factor of ten
    scale *= 10.0
  elif acc_rate > 0.75:
    # increase by double
    scale *= 2.0
  elif acc_rate > 0.5:
    # increase by ten percent
    scale *= 1.1

  return scale


def delta_logp(model, q, q0):
  """ Function for calculating acceptance criteria """
  val = model.logp(q) - model.logp(q0)
  val = min(1, np.exp(val))

  return val
