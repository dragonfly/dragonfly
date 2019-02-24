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
from .continuous import Normal
from .model import Model
from ..utils.base_test_class import BaseTestClass, execute_tests

class PosteriorTestCase(BaseTestClass):
  """ Unit test for posterior sampling with Gaussian likelihood and Gaussian prior"""

  def setUp(self):
    """ Sets up unit tests. """
    self.size = 1000
    self.threshold = 0.01
    self.data = np.random.normal(5, 2, 1000)
    self.m = 0
    self.tau = 4
    self.prior = Normal(self.m, self.tau)
    self.var = 2

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
    dist = Normal(x, self.var)
    val = self.prior.pdf(x)
    for i in range(len(self.data)):
      val = val*dist.pdf(self.data[i])
    return val

  def _logp(self, x):
    """ Compute log pdf of model """
    dist = Normal(x, self.var)
    val = self.prior.logp(x)
    for i in range(len(self.data)):
      val = val + dist.logp(self.data[i])
    return val

  def _grad_logp(self, x):
    """ Computes gradient of log pdf of model """
    dist = Normal(x, self.var)
    val = self.prior.grad_logp(x)
    for i in range(len(self.data)):
      val = val - dist.grad_logp(self.data[i])
    return val

  @unittest.skip
  def test_posterior_NUTS(self):
    """ Tests posterior estimation using NUTS sampling """
    self.report('Test posterior estimation with Gaussian likelihood and Gaussian prior '
                'using NUTS sampling')

    model_a = Model(self._pdf, self._logp, self._grad_logp)
    samples = model_a.draw_samples('nuts', self.size, np.array([3]), 10)

    avg = 0
    for i in range(len(self.data)):
      avg = avg + self.data[i]
    avg = avg/len(self.data)

    # Map estimation
    val = self.var/len(self.data)
    mean_a = (self.tau*avg)/(self.tau + val) + self.m*(val/(self.tau + val))
    var_a = (val*self.tau)/(self.tau + val)

    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)

    self._check_sample_sizes(samples)
    assert abs(mean_a - mean_r) <= self.threshold
    assert abs(var_a - var_r) <= self.threshold
    self.report('Map Estimation: mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(mean_a, var_a, mean_r, var_r), 'test_result')

  @unittest.skip
  def test_posterior_Slice(self):
    """ Tests posterior estimation using Slice sampling """
    self.report('Test posterior estimation with Gaussian likelihood and Gaussian prior '
                'using Slice sampling')

    model_a = Model(self._pdf, self._logp, self._grad_logp)
    samples = model_a.draw_samples('slice', self.size, np.array([4.5]))

    avg = 0
    for i in range(len(self.data)):
      avg = avg + self.data[i]
    avg = avg/len(self.data)

    # Map estimation
    val = self.var/len(self.data)
    mean_a = (self.tau*avg)/(self.tau + val) + self.m*(val/(self.tau + val))
    var_a = (val*self.tau)/(self.tau + val)

    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)

    self._check_sample_sizes(samples)
    assert abs(mean_a - mean_r) <= self.threshold
    assert abs(var_a - var_r) <= self.threshold
    self.report('Map Estimation: mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(mean_a, var_a, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()
