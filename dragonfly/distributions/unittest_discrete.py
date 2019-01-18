"""
  Unit tests for discrete distributions.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import

# pylint: disable=invalid-name
# pylint: disable=no-self-use

import numpy as np

# Local imports
from .discrete import Bernoulli, Binomial, DiscreteUniform
from ..utils.base_test_class import BaseTestClass, execute_tests

class DiscreteDistributionsTestCase(BaseTestClass):
  """ Unit tests for distributions in discrete.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.size = (10000, 10000)
    self.threshold = 0.01

  def _check_sample_sizes(self, samples):
    """ Compares the sample sizes with the size parameter"""
    assert self.size == samples.shape

  def _compute_mean(self, samples):
    """ Computes Mean """
    return np.mean(samples)

  def _compute_variance(self, samples):
    """ Computes Variance """
    return np.var(samples)

  def test_random_sampling_bernoulli(self):
    """ Tests random sampling from Bernoulli distribution """
    self.report('Test random sampling of Bernoulli Distribution.')
    p = 0.6
    dist = Bernoulli(p)
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

  def test_random_sampling_binomial(self):
    """ Tests random sampling from Binomial distribution """
    self.report('Test random sampling of Binomial Distribution.')
    n = 10
    p = 0.6
    dist = Binomial(n, p)
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

  def test_random_sampling_discreteuniform(self):
    """ Tests random sampling from Discrete Uniform distribution """
    self.report('Test random sampling of Discrete Uniform Distribution.')
    lower = -5
    upper = 5
    dist = DiscreteUniform(lower, upper)
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

