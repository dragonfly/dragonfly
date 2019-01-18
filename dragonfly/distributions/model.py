"""
  Class for abstract probability distribution
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
# pylint: disable=invalid-name
# pylint: disable=abstract-method
# pylint: disable=relative-import

import numpy as np

# Local imports
from .distribution import Distribution
from ..sampling.slice import Slice
from ..sampling.nuts import NoUTurnSamplerDA as NUTS, LeapFrog
from ..sampling.metropolis import Metropolis, BinaryMetropolis

class Model(Distribution):
  """ Class for abstract distributions """

  def __init__(self, pdf, logp, grad_logp):
    super(Model, self).__init__()
    self._pdf = pdf
    self._logp = logp
    self._grad_logp = grad_logp
    self.dim = None

  def pdf(self, x):
    """ Returns pdf of distribution at x """
    return np.asscalar(self._pdf(x))

  def logp(self, x):
    """ Returns log of pdf at x """
    return self._logp(x)

  def grad_logp(self, x):
    """ Returns gradient of log pdf at x """
    return self._grad_logp(x)

  def draw_samples(self, method, size=None, init_sample=None, *args):
    """ Returns samples drawn from distribution. """
    if method == 'nuts':
      return self.draw_samples_nuts(size, init_sample, *args)
    elif method == 'slice':
      return self.draw_samples_slice(size, init_sample, *args)
    elif method == 'metropolis':
      return self.draw_samples_metropolis(size, init_sample, *args)
    elif method == 'binarymetropolis':
      return self.draw_samples_binary_metropolis(size, init_sample, *args)

  def draw_samples_slice(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Slice sampler"""
    sampler = Slice(self)
    return sampler.sample(init_sample, size, *args)

  def draw_samples_nuts(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using NUTS sampler"""
    sampler = NUTS(self, LeapFrog)
    return sampler.sample(init_sample, size, *args)

  def draw_samples_metropolis(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Metropolis sampler"""
    if hasattr(init_sample, '__len__'):
      self.dim = len(init_sample)
    else:
      self.dim = 1
    sampler = Metropolis(self, True, *args)
    return sampler.sample(init_sample, size)

  def draw_samples_binary_metropolis(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Binary Metropolis sampler"""
    sampler = BinaryMetropolis(self, True, *args)
    return sampler.sample(init_sample, size)
