"""
  Class for joint probability distribution of independent random variables
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
# pylint: disable=invalid-name
# pylint: disable=abstract-method

import numpy as np

# Local imports
from .distribution import Distribution

class JointDistribution(Distribution):
  """ Independent Joint Distribution """

  def __init__(self, rv_s):
    super(JointDistribution, self).__init__()
    self.rv_s = rv_s

  def get_rv_s(self):
    """ Returns the list of random variables """
    return self.rv_s

  def draw_samples(self, method, size=None, init_sample=None, *args):
    """ Returns samples drawn from distribution. """
    if method == 'random':
      return self.draw_random_samples(size)
    elif method == 'nuts':
      return self.draw_samples_nuts(size, init_sample, *args)
    elif method == 'slice':
      return self.draw_samples_slice(size, init_sample, *args)
    elif method == 'metropolis':
      return self.draw_samples_metropolis(size, init_sample, *args)

  def draw_random_samples(self, size=None):
    """ Returns random samples drawn from distribution. """
    samples = np.zeros((len(self.rv_s), size))
    for i, rv in enumerate(self.rv_s):
      samples[i] = rv.draw_samples('random', size)
    return samples

  def draw_samples_slice(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Slice sampler"""
    samples = np.zeros((len(self.rv_s), size))
    for i, rv in enumerate(self.rv_s):
      samples[i] = rv.draw_samples('slice', size, init_sample, *args)
    return samples

  def draw_samples_nuts(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using NUTS sampler"""
    samples = np.zeros((len(self.rv_s), size))
    for i, rv in enumerate(self.rv_s):
      samples[i] = rv.draw_samples('nuts', size, init_sample, *args)
    return samples

  def draw_samples_metropolis(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Metropolis sampler"""
    samples = np.zeros((len(self.rv_s), size))
    for i, rv in enumerate(self.rv_s):
      samples[i] = rv.draw_samples('metropolis', size, init_sample, *args)
    return samples
