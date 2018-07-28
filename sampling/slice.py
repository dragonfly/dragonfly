"""
  Slice Sampler -- Adapted from pymc3 library: https://github.com/pymc-devs/pymc3
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import division

# pylint: disable=invalid-name

import numpy as np
import numpy.random as nr

__all__ = ['Slice']

LOOP_ERR_MSG = 'max slicer iters %d exceeded'

class Slice(object):
  """
  Class for Slice sampler.

  Parameters
  ----------
  model : Distribution from which sampling has done
  w     : Initial width for slice
  tune  : Flag for tuning
  """

  def __init__(self, model, w=1., tune=True, iter_limit=np.inf):
    """
    Constructor for slice sampler.
    """
    self.model = model
    self.w = w
    self.tune = tune
    self.n_tunes = 0.
    self.iter_limit = iter_limit

  def _sample(self, q0):
    """
    Helper function which implements slice sampler.
    """
    self.w = np.resize(self.w, len(q0))  # this is a repmat
    q = np.copy(q0)
    ql = np.copy(q0)  # l for left boundary
    qr = np.copy(q0)  # r for right boudary
    for i, _ in enumerate(q0):
      # uniformly sample from 0 to p(q), but in log space
      y = self.model.logp(q) - nr.standard_exponential()
      ql[i] = q[i] - nr.uniform(0, self.w[i])
      qr[i] = q[i] + self.w[i]

      # Stepping out procedure
      cnt = 0
      while y < self.model.logp(ql):
        ql[i] -= self.w[i]
        cnt += 1
        if cnt > self.iter_limit:
          raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)
      cnt = 0
      while y < self.model.logp(qr):
        qr[i] += self.w[i]
        cnt += 1
        if cnt > self.iter_limit:
          raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)

      cnt = 0
      q[i] = (qr[i] - ql[i])*nr.rand() + ql[i]
      while self.model.logp(q) < y:
        # Sample uniformly from slice
        if q[i] > q0[i]:
          qr[i] = q[i]
        elif q[i] < q0[i]:
          ql[i] = q[i]
        q[i] = (qr[i] - ql[i])*nr.rand() + ql[i]
        cnt += 1
        if cnt > self.iter_limit:
          raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)

      if self.tune:
        self.w[i] = self.w[i] * (self.n_tunes / (self.n_tunes + 1)) +\
          (qr[i] - ql[i]) / (self.n_tunes + 1)  # same as before
        qr[i] = q[i]
        ql[i] = q[i]

    if self.tune:
      self.n_tunes += 1

    return q

  def sample(self, q0, num_samples=1, burn=100):
    """
    sample -- function which populates samples
    by calling function _sample iteratively
    """
    if num_samples is None:
      num_samples = 1
    if not hasattr(q0, '__len__'):
      q0 = np.array([q0])

    # Burn in samples
    for i in range(burn):
      q0 = self._sample(q0)

    samples = np.zeros([num_samples, len(q0)])
    for i in range(num_samples):
      samples[i] = self._sample(q0)
      q0 = samples[i]

    return samples
