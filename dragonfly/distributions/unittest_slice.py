"""
  Unit tests for Slice sampling.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
import unittest

# pylint: disable=no-self-use

import numpy as np

# Local imports
from .continuous import Normal, Exponential, ContinuousUniform, Beta
from ..utils.base_test_class import BaseTestClass, execute_tests

class SliceTestCase(BaseTestClass):
  """ Unit tests for slice sampling """

  def setUp(self):
    """ Sets up unit tests. """
    self.size = 100000
    self.threshold = 0.05

  def _check_sample_sizes(self, samples):
    """ Compares the sample sizes with the size parameter"""
    assert self.size == len(samples)

  def _compute_mean(self, samples):
    """ Computes Mean """
    return np.mean(samples)

  def _compute_variance(self, samples):
    """ Computes Variance """
    return np.var(samples)

  @unittest.skip
  def test_slice_sampling_normal(self):
    """ Tests slice sampling from Normal distribution """
    self.report('Test slice sampling from Normal Distribution.')
    mean = 11
    var = 3
    dist = Normal(mean, var)
    samples = dist.draw_samples('slice', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s :: test result: mean=%0.3f, variance=%0.3f'\
                 %(str(dist), mean_r, var_r), 'test_result')

  def test_slice_sampling_exponential(self):
    """ Tests slice sampling from Exponential distribution """
    self.report('Test slice sampling of Exponential Distribution.')
    lam = 2
    dist = Exponential(lam)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('slice', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= 0).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  @unittest.skip
  def test_slice_sampling_uniform(self):
    """ Tests slice sampling from Continuous Uniform """
    self.report('Test slice sampling of Continuous Uniform Distribution.')
    lower = -5
    upper = 5
    dist = ContinuousUniform(lower, upper)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('slice', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= lower).all() and (samples <= upper).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  def test_slice_sampling_beta(self):
    """ Tests slice sampling from Beta Distribution """
    self.report('Test slice sampling of Beta Distribution.')
    alpha = 1
    beta = 2
    dist = Beta(alpha, beta)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('slice', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= 0).all() and (samples <= 1).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()
