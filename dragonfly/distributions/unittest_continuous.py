"""
  Unit tests for continuous distributions.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
import unittest
# pylint: disable=no-self-use

import numpy as np
import warnings

# Local imports
from .continuous import Beta, ContinuousUniform, Exponential, \
                                     Normal, MultivariateGaussian
from ..utils.base_test_class import BaseTestClass, execute_tests

class ContinuousDistributionsTestCase(BaseTestClass):
  """ Unit tests for distributions in continuous.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.size = 1000000
    self.threshold = 0.01
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")

  def _check_sample_sizes(self, samples):
    """ Compares the sample sizes with the size parameter"""
    assert self.size == len(samples)

  def _compute_mean(self, samples, axis=None):
    """ Computes Mean """
    return np.mean(samples, axis)

  def _compute_variance(self, samples, axis=None):
    """ Computes Variance """
    return np.var(samples, axis)

  def _compute_covariance(self, samples):
    """ Computes Covariance """
    return np.cov(samples.T)

  def test_rand_sampling_normal(self):
    """ Tests random sampling from Normal distribution """
    self.report('Test random sampling of Normal Distribution.')
    mean = 0
    variance = 1
    dist = Normal(mean, variance)
    samples = dist.draw_samples('random', self.size)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(variance - var_r) <= self.threshold
    self.report('%s :: test result: mean=%0.3f, variance=%0.3f'%\
                (str(dist), mean_r, var_r), 'test_result')

  def test_rand_sampling_multi_normal(self):
    """ Tests random sampling from Multivariate Normal distribution """
    self.report('Test random sampling of Multivariate Normal Distribution.')
    cov_thresh = 0.1
    mean_thresh = 0.01
    mean = np.arange(3)
    covariance = 3*np.identity(3)
    dist = MultivariateGaussian(mean, covariance)
    samples = dist.draw_samples('random', self.size)
    mean_r = self._compute_mean(samples, 0)
    self._check_sample_sizes(samples)
    assert (abs(mean - self._compute_mean(samples, 0)) <= mean_thresh).all()
    assert (abs(covariance - self._compute_covariance(samples)) <= cov_thresh).all()
    self.report('%s :: test result: mean=%s'%(str(dist), str(mean_r)), 'test_result')

  def test_rand_sampling_expo(self):
    """ Tests random sampling from Exponential distribution """
    self.report('Test random sampling of Exponential Distribution.')
    lam = 2
    dist = Exponential(lam)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('random', self.size)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  @unittest.skip("Stochastic")
  def test_rand_sampling_uniform(self):
    """ Tests random sampling from Continuous Uniform """
    self.report('Test random sampling of Continuous Uniform Distribution.')
    lower = -5
    upper = 5
    dist = ContinuousUniform(lower, upper)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('random', self.size)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  def test_rand_sampling_beta(self):
    """ Tests random sampling from Beta Distribution """
    self.report('Test random sampling of Beta Distribution.')
    alpha = 1
    beta = 2
    dist = Beta(alpha, beta)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('random', self.size)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()
