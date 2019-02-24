"""
  Unit tests for NUTS sampling.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import

# pylint: disable=no-self-use
import unittest

import numpy as np

# Local imports
from .continuous import Normal, Beta, Exponential
from ..utils.base_test_class import BaseTestClass, execute_tests


@unittest.skip
class NutsTestCase(BaseTestClass):
  """ Unit tests for distributions in continuous.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.size = 100000
    self.threshold = 0.1

  def _check_sample_sizes(self, samples):
    """ Compares the sample sizes with the size parameter"""
    assert self.size == len(samples)

  def _compute_mean(self, samples):
    """ Computes Mean """
    return np.mean(samples)

  def _compute_variance(self, samples):
    """ Computes Variance """
    return np.var(samples)

  def test_nuts_sampling_normal(self):
    """ Tests NUTS sampling from Normal distribution """
    self.report('Test NUTS sampling of Normal Distribution.')
    mean = 11
    variance = 3
    dist = Normal(mean, variance)
    samples = dist.draw_samples('nuts', self.size, np.array([0.1]), 10)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(variance - var_r) <= self.threshold
    self.report('%s :: test result: mean=%0.3f, variance=%0.3f'\
                 %(str(dist), mean_r, var_r), 'test_result')

  def test_nuts_sampling_beta(self):
    """ Tests nuts sampling from Beta Distribution """
    self.report('Test nuts sampling of Beta Distribution.')
    alpha = 1
    beta = 2
    dist = Beta(alpha, beta)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('nuts', self.size, np.array([0.1]), 10)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= 0).all() and (samples <= 1).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  def test_nuts_sampling_exponential(self):
    """ Tests nuts sampling from Exponential distribution """
    self.report('Test nuts sampling of Exponential Distribution.')
    lam = 2
    dist = Exponential(lam)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('nuts', self.size, np.array([0.1]), 10)
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= 0).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()

