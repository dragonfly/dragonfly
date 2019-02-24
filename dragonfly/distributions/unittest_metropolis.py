"""
  Unit tests for metropolis sampling of continuous distributions.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
import unittest

# pylint: disable=no-self-use

import numpy as np

# Local imports
from .continuous import Normal, Exponential, ContinuousUniform
from ..utils.base_test_class import BaseTestClass, execute_tests

class MetropolisTestCases(BaseTestClass):
  """ Unit tests for metropolis sampling """

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

  def test_metropolis_normal(self):
    """ Tests metropolis sampling from Normal distribution """
    self.report('Test metropolis sampling from Normal Distribution.')
    mean = 11
    var = 3
    dist = Normal(mean, var)
    samples = dist.draw_samples('metropolis', self.size, np.array([11]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s :: test result: mean=%0.3f, variance=%0.3f'\
                 %(str(dist), mean_r, var_r), 'test_result')

  def test_metropolis_exponential(self):
    """ Tests metropolis sampling from Exponential distribution """
    self.report('Test metropolis sampling of Exponential Distribution.')
    lam = 2
    dist = Exponential(lam)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('metropolis', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= 0).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  @unittest.skip('Stochastic')
  def test_metropolis_uniform(self):
    """ Tests metropolis sampling from Continuous Uniform """
    self.report('Test metropolis sampling of Continuous Uniform Distribution.')
    lower = -5
    upper = 5
    dist = ContinuousUniform(lower, upper)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('metropolis', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= lower).all() and (samples <= upper).all()
    assert abs(mean - mean_r) <= self.threshold
    assert abs(var - var_r) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()
