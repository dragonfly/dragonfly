"""
  Unit tests for posterior sampling
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
from __future__ import division
import unittest

# pylint: disable=invalid-name
# pylint: disable=no-self-use

import numpy as np

# Local imports
from .continuous import Beta
from .model import Model
from ..utils.base_test_class import BaseTestClass, execute_tests

class PosteriorMixedTestCase(BaseTestClass):
  """ Unit test for posterior sampling with Bernoulli likelihood and Beta prior"""

  def setUp(self):
    """ Sets up unit tests. """
    self.size = 10000
    self.n = 1000
    self.threshold = 0.1
    self.alpha = 2
    self.beta = 3
    self.data = np.random.binomial(1, 0.6, self.n)
    self.Y = np.sum(self.data)
    self.prior = Beta(self.alpha, self.beta)

  def _check_sample_sizes(self, samples):
    """ Compares the sample sizes with the size parameter"""
    assert self.size == len(samples)

  def _compute_mean(self, samples):
    """ Computes Mean """
    return np.mean(samples)

  def _compute_variance(self, samples):
    """ Computes Variance """
    return np.var(samples)

  def _pdf(self, x):
    """ Computes pdf of model """
    if x < 0 or x > 1:
      return 0
    value = np.power(x, self.Y + self.alpha - 1)
    value = value*np.power(1 - x, self.n - self.Y + self.beta - 1)
    return value

  def _logp(self, x):
    """ Compute log pdf of model """
    if x < 0 or x > 1:
      return -np.inf
    value = (self.Y + self.alpha - 1)*np.log(x)
    value = value + (self.n - self.Y + self.beta - 1)*np.log(1 - x)
    return value

  def _grad_logp(self, x):
    """ Computes gradient of log pdf of model """
    if x < 0 or x > 1:
      return 0
    value = (self.Y + self.alpha - 1.0)/x
    value = value + (self.n - self.Y + self.beta - 1.0)/(1 - x)
    return value

  @unittest.skip
  def test_posterior_NUTS(self):
    """ Tests posterior estimation using NUTS sampling """
    self.report('Test posterior estimation with Bernoulli likelihood and Beta prior '
                'using NUTS sampling')

    model_a = Model(self._pdf, self._logp, self._grad_logp)
    samples = model_a.draw_samples('nuts', self.size, np.array([0.1]), 10)

    # Map estimation
    alpha_a = self.Y + self.alpha
    beta_a = self.n - self.Y + self.beta
    mean_a = alpha_a/float(alpha_a + beta_a)
    var_a = (alpha_a*beta_a)/((np.power(alpha_a+beta_a, 2))*(alpha_a+beta_a+1))

    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)

    self._check_sample_sizes(samples)
    assert abs(mean_a - mean_r) <= self.threshold
    assert abs(var_a - var_r) <= self.threshold
    self.report('Map Estimation: mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(mean_a, var_a, mean_r, var_r), 'test_result')

  def test_posterior_Slice(self):
    """ Tests posterior estimation using Slice sampling """
    self.report('Test posterior estimation with Bernoulli likelihood and Beta prior '
                'using Slice sampling')

    model_a = Model(self._pdf, self._logp, self._grad_logp)
    samples = model_a.draw_samples('slice', self.size, np.array([0.1]))

    # Map estimation
    alpha_a = self.Y + self.alpha
    beta_a = self.n - self.Y + self.beta
    mean_a = alpha_a/float(alpha_a + beta_a)
    var_a = (alpha_a*beta_a)/((np.power(alpha_a+beta_a, 2))*(alpha_a+beta_a+1))

    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)

    self._check_sample_sizes(samples)
    assert abs(mean_a - mean_r) <= self.threshold
    assert abs(var_a - var_r) <= self.threshold
    self.report('Map Estimation: mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(mean_a, var_a, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()
