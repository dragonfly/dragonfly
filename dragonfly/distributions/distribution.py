"""
  Base Classes for probability distributions
  -- kvysyara@andrew.cmu.edu
"""

# pylint: disable=invalid-name

# Local imports
from ..sampling.slice import Slice
from ..sampling.metropolis import Metropolis, BinaryMetropolis
from ..sampling.nuts import NoUTurnSamplerDA as NUTS, LeapFrog


class Distribution(object):
  """ An abstract class which implements probability distributions """

  def __init__(self):
    """ Constructor. """
    self.domain = None
    self.dim = None
    super(Distribution, self).__init__()

  def draw_samples(self, method, *args):
    """ Returns random samples drawn from distribution. """
    raise NotImplementedError('Implement in a child class.')

  def draw_samples_random(self, method, *args):
    """ Returns random samples drawn from distribution. """
    raise NotImplementedError('Implement in a child class.')

  def get_mean(self):
    """ Returns the mean of the distribution. """
    raise NotImplementedError('Implement in a child class.')

  def get_variance(self):
    """ Returns the variance of the distribution. """
    raise NotImplementedError('Implement in a child class.')

  def get_domain(self):
    """ Returns domain """
    return self.domain

  def get_dim(self):
    """ Returns dimension """
    return self.dim


class Discrete(Distribution):
  """ Base class for Discrete distributions """

  def __init__(self):
    """ Constructor. """
    super(Discrete, self).__init__()

  def pmf(self, x):
    """ Returns pmf of distribution at x """
    raise NotImplementedError('Implement in a child class.')

  def draw_samples(self, method, size=None, init_sample=None, *args):
    """ Returns samples drawn from distribution. """
    if method == 'random':
      return self.draw_samples_random(size)
    elif method == 'metropolis':
      return self.draw_samples_metropolis(size, init_sample, *args)
    elif method == 'binarymetropolis':
      return self.draw_samples_binary_metropolis(size, init_sample, *args)

  def draw_samples_metropolis(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Metropolis sampler"""
    sampler = Metropolis(self, True, *args)
    return sampler.sample(init_sample, size)

  def draw_samples_binary_metropolis(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Binary Metropolis sampler"""
    sampler = BinaryMetropolis(self, True, *args)
    return sampler.sample(init_sample, size)


class Continuous(Distribution):
  """ Base class for Continuous distributions """

  def __init__(self):
    """ Constructor. """
    super(Continuous, self).__init__()

  def pdf(self, x):
    """ Returns pdf of distribution at x """
    raise NotImplementedError('Implement in a child class.')

  def draw_samples(self, method, size=None, init_sample=None, *args):
    """ Returns samples drawn from distribution. """
    if method == 'random':
      return self.draw_samples_random(size)
    elif method == 'nuts':
      return self.draw_samples_nuts(size, init_sample, *args)
    elif method == 'slice':
      return self.draw_samples_slice(size, init_sample, *args)
    elif method == 'metropolis':
      return self.draw_samples_metropolis(size, init_sample, *args)

  def draw_samples_slice(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Slice sampler"""
    sampler = Slice(self, *args)
    return sampler.sample(init_sample, size)

  def draw_samples_nuts(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using NUTS sampler"""
    sampler = NUTS(self, LeapFrog)
    return sampler.sample(init_sample, size, *args)

  def draw_samples_metropolis(self, size, init_sample, *args):
    """ Returns samples drawn from distribution using Metropolis sampler"""
    sampler = Metropolis(self, *args)
    return sampler.sample(init_sample, size)

