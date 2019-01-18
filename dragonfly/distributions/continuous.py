"""
  Classes for various continuous probability distributions
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
from __future__ import division

# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=no-self-use

import numpy as np

# Local imports
from .distribution import Continuous
from ..exd import domains

# Univariate Normal Distribution
class Normal(Continuous):
  """ Univariate Normal Distribution """

  def __init__(self, mean, var):
    """ Constructor. """
    super(Normal, self).__init__()
    self.mean = float(mean)
    self.var = float(var)
    self.dim = 1
    self.domain = domains.EuclideanDomain([[-np.inf, np.inf]])

  def pdf(self, x):
    """ Returns value of pdf at x """
    return np.asscalar(np.exp(-((x-self.mean)**2)/(2*self.var))/\
                       (np.sqrt(2*np.pi*self.var)))

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    return np.random.normal(self.mean, np.sqrt(self.var), size)

  def logp(self, x):
    """ Returns the log pdf at x """
    return np.asscalar(-0.5*(((x-self.mean)*(x-self.mean)/self.var)\
                        + np.log(2*np.pi*self.var)))

  def grad_logp(self, x):
    """ Returns gradient of log pdf at x """
    return np.asscalar(-(x - self.mean)/self.var)

  def get_mean(self):
    """ Returns mean. """
    return self.mean

  def get_variance(self):
    """ Returns variance. """
    return self.var

  def __str__(self):
    """ Returns a string representation. """
    return 'Univariate Normal: mean=%0.3f, variance=%.3f' % (self.mean, self.var)


# Multivariate Gaussian Distribution
class MultivariateGaussian(Continuous):
  """ Multivariate Gaussian Distribution """

  def __init__(self, mean, cov):
    """ Constructor. """
    super(MultivariateGaussian, self).__init__()
    self.mean = np.array(mean, dtype=float)
    self.cov = np.array(cov, dtype=float)
    self.pre = np.linalg.inv(self.cov)
    self.det = np.linalg.det(self.cov)
    self.dim = len(mean)
    self.domain = domains.EuclideanDomain(np.tile(np.array([-np.inf, np.inf]),\
                                          (len(mean), 1)))

  def pdf(self, x):
    """ Returns value of pdf at x """
    value = -0.5*np.dot(np.transpose(x - self.mean), np.dot(self.pre, x - self.mean))
    return np.asscalar(np.exp(value)*np.power(2*np.pi*self.det, -0.5))

  def draw_samples_random(self, size):
    """ Returns random samples drawn from distribution. """
    return np.random.multivariate_normal(self.mean, self.cov, size)

  def logp(self, x):
    """ Returns the log pdf at x """
    value = np.dot(np.transpose(x - self.mean), np.dot(self.pre, x - self.mean))
    return -0.5*(value + np.log(2*np.pi) + np.log(self.det))

  def grad_logp(self, x):
    """ Returns gradient of log pdf at x """
    return -np.dot(self.pre, x - self.mean)

  def get_mean(self):
    """ Returns mean. """
    return self.mean

  def get_variance(self):
    """ Returns variance. """
    return self.cov

  def __str__(self):
    """ Returns a string representation. """
    return 'Multivariate Normal'


# Continuous Uniform Distribution
class ContinuousUniform(Continuous):
  """ Continuous Uniform Distribution """

  def __init__(self, lower, upper):
    """ Constructor. """
    super(ContinuousUniform, self).__init__()
    self.lower = float(lower)
    self.upper = float(upper)
    self.dim = 1
    self.domain = domains.EuclideanDomain([[self.lower, self.upper]])

  def pdf(self, x):
    """ Returns value of pdf at x """
    if x < self.lower or x > self.upper:
      return 0
    return np.asscalar(1/(self.upper - self.lower))

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    return np.random.uniform(self.lower, self.upper, size)

  def logp(self, x):
    """ Returns the log pdf at x """
    if x < self.lower or x > self.upper:
      return -np.inf
    return -np.log(self.upper - self.lower)

  def grad_logp(self, x):
    """ Returns gradient of log pdf at x """
    return 0

  def get_parameters(self):
    """ Returns parameters """
    return self.lower, self.upper

  def get_mean(self):
    """ Returns mean """
    return (self.lower + self.upper)/2

  def get_variance(self):
    """ Returns variance """
    return (np.power((self.upper - self.lower), 2))/12

  def __str__(self):
    """ Returns a string representation. """
    return 'Continuous Uniform: lower=%0.3f, upper=%.3f' % (self.lower, self.upper)


# Exponential Distribution
class Exponential(Continuous):
  """ Exponential Distribution """

  def __init__(self, lam):
    """ Constructor. """
    super(Exponential, self).__init__()
    self.lam = float(lam)
    self.dim = 1
    self.domain = domains.EuclideanDomain([[0, np.inf]])

  def pdf(self, x):
    """ Returns value of pdf at x """
    if x < 0:
      return 0
    return np.asscalar(self.lam*np.exp(-self.lam*x))

  def draw_samples_random(self, size=None):
    """ Returns random samples drawn from distribution. """
    return np.random.exponential(1/self.lam, size)

  def logp(self, x):
    """ Returns the log pdf at x """
    if x < 0:
      return -np.inf
    return np.log(self.lam) - self.lam*x

  def grad_logp(self, x):
    """ Returns gradient of log pdf at x """
    return -self.lam

  def get_lambda(self):
    """ Returns lambda parameter"""
    return self.lam

  def get_mean(self):
    """ Returns mean """
    return 1/self.lam

  def get_variance(self):
    """ Returns variance """
    return np.power(self.lam, -2)

  def __str__(self):
    """ Returns a string representation. """
    return 'Exponential: lambda=%0.3f' % (self.lam)


# Beta Distribution
class Beta(Continuous):
  """ Beta Distribution """

  def __init__(self, alpha, beta):
    """ Constructor. """
    super(Beta, self).__init__()
    self.alpha = float(alpha)
    self.beta = float(beta)
    self.dim = 1
    self.B = (np.math.factorial(self.alpha - 1)*np.math.factorial(self.beta - 1))/\
                  (float(np.math.factorial(self.alpha+self.beta-1)))
    self.domain = domains.EuclideanDomain([[0, 1]])

  def pdf(self, x):
    """ Returns value of pdf at x """
    if x < 0 or x > 1:
      return 0
    return np.asscalar((np.power(x, self.alpha - 1)*\
                        np.power(1 - x, self.beta - 1))/self.B)

  def draw_samples_random(self, *args):
    """ Returns random samples drawn from distribution. """
    return np.random.beta(self.alpha, self.beta, *args)

  def logp(self, x):
    """ Returns the log pdf at x """
    if x < 0 or x > 1:
      return -np.inf
    return (self.alpha - 1)*np.log(x) + (self.beta - 1)*np.log(1-x) - np.log(self.B)

  def grad_logp(self, x):
    """ Returns gradient of log pdf at x """
    if x < 0 or x > 1:
      return 0
    return (self.alpha - 1)/x - (self.beta - 1)/(1 - x)

  def get_parameters(self):
    """ Returns parameters """
    return self.alpha, self.beta

  def get_mean(self):
    """ Returns mean """
    return self.alpha/(self.alpha + self.beta)

  def get_variance(self):
    """ Returns variance """
    return (self.alpha*self.beta)/\
            ((np.power(self.alpha+self.beta, 2))*(self.alpha+self.beta+1))

  def __str__(self):
    """ Returns a string representation. """
    return 'Beta: alpha=%0.3f, beta=%.3f' % (self.alpha, self.beta)
