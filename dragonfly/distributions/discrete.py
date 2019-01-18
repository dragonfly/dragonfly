"""
  Classes for various discrete probability distributions
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
# pylint: disable=invalid-name
# pylint: disable=abstract-method
# pylint: disable=relative-import

import numpy as np

# Local imports
from .distribution import Discrete
from ..exd import domains

# Bernoulli Distribution
class Bernoulli(Discrete):
  """ Bernoulli Distribution """

  def __init__(self, p):
    """ Constructor. """
    super(Bernoulli, self).__init__()
    self.p = float(p)
    self.dim = 1
    self.domain = domains.IntegralDomain([[0, 1]])

  def pmf(self, x):
    """ Returns pmf of distribution at x. """
    return np.asscalar(self.p*x + (1 - x)*(1 - self.p))

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    return np.random.binomial(1, self.p, size)

  def get_parameters(self):
    """ Returns parameters """
    return self.p

  def logp(self, x):
    """ Returns log probability of x """
    if x < 0 or x > 1:
      return -np.inf
    return x*np.log(self.p) + (1-x)*np.log(1-self.p)

  def get_mean(self):
    """ Returns mean """
    return self.p

  def get_variance(self):
    """ Returns variance """
    return self.p*(1 - self.p)

  def __str__(self):
    """ Returns a string representation. """
    return 'Bernoulli: p=%0.3f' % (self.p)


# Binomial Distribution
class Binomial(Discrete):
  """ Binomial Distribution """

  def __init__(self, n, p):
    """ Constructor. """
    super(Binomial, self).__init__()
    self.n = n
    self.p = float(p)
    self.dim = 1
    self.domain = domains.IntegralDomain([[0, self.n]])

  def pmf(self, k):
    """ Returns pmf of distribution at x. """
    value = np.math.factorial(self.n)
    value = value/float((np.math.factorial(k)*np.math.factorial(self.n-k)))
    return np.asscalar(value*np.power(self.p, k)*np.power(1-self.p, self.n-k))

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    return np.random.binomial(self.n, self.p, size)

  def get_parameters(self):
    """ Returns parameters """
    return self.n, self.p

  def logp(self, x):
    """ Returns log probability of x """
    if x < 0 or x > self.n:
      return -np.inf

    value = np.math.factorial(self.n)
    value = value/float(np.math.factorial(x)*np.math.factorial(self.n-x))
    return np.log(value) + x*np.log(self.p) + (self.n - x)*np.log(1-self.p)

  def get_mean(self):
    """ Returns mean """
    return self.n*self.p

  def get_variance(self):
    """ Returns variance """
    return self.n*self.p*(1 - self.p)

  def __str__(self):
    """ Returns a string representation. """
    return 'Binomial: n=%0.3f, p=%0.3f' % (self.n, self.p)


# Categorical Distribution
class Categorical(Discrete):
  """ Categorical Distribution """

  def __init__(self, categories, p):
    """ Constructor """
    super(Categorical, self).__init__()
    self.cat = list(categories)
    self.k = len(categories)
    self.p = np.array(p)
    self.dim = 1
    self.domain = domains.IntegralDomain([[1, self.k]])

  def pmf(self, i):
    """ Returns pmf of distribution at x. """
    if i < 0 or i >= self.k:
      return 0
    return np.asscalar(self.p[i])

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    samples = np.random.multinomial(1, self.p, size)
    return np.argmax(samples, len(samples.shape) - 1)

  def get_category(self, i):
    """ Returns category value. """
    if i < 0 or i >= self.k:
      return None
    return self.cat[int(i)]

  def get_parameters(self):
    """ Returns parameters """
    return self.cat, self.p

  def logp(self, value):
    """ Returns log probability of category """
    if hasattr(value, '__len__'):
      if len(value) == 1:
        value = np.asscalar(value)
      else:
        raise ValueError('Input dimension should be 1.')

    if value < 0 or value >= self.k:
      return -np.inf
    return np.log(self.p[value])

  def get_id(self, category):
    """Returns id for the category. """
    if category is None or np.isnan(category):
      return -1
    return self.cat.index(category)

  def __str__(self):
    """ Returns a string representation. """
    return 'Categorical: k=%0.3f, p=%0.3f' % (self.k, self.p)


# Discrete Uniform Discrete
class DiscreteUniform(Discrete):
  """ Discrete Uniform Distribution """

  def __init__(self, lower, upper):
    """ Constructor. """
    super(DiscreteUniform, self).__init__()
    self.lower = np.floor(lower)
    self.upper = np.floor(upper)
    self.dim = 1
    self.domain = domains.IntegralDomain([[self.lower, self.upper]])

  def pmf(self, x):
    """ Returns pmf of distribution at x. """
    if x >= self.lower and x <= self.upper:
      return 1/(self.upper - self.lower + 1)
    return 0

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    return np.random.randint(self.lower, self.upper+1, size)

  def get_parameters(self):
    """ Returns parameters """
    return self.lower, self.upper

  def logp(self, x):
    """ Returns log probability of x """
    if x < self.lower or x > self.upper:
      return -np.inf

    return -np.log(self.upper - self.lower + 1)

  def get_mean(self):
    """ Returns mean """
    return (self.lower + self.upper)/2

  def get_variance(self):
    """ Returns variance """
    return (np.power((self.upper - self.lower + 1), 2) - 1)/12

  def __str__(self):
    """ Returns a string representation. """
    return 'Discrete Uniform: lower=%0.3f, upper=%0.3f' % (self.lower, self.upper)

