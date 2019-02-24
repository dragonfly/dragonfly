"""
  Unit tests for metropolis sampling of discrete distributions.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import
import unittest
# pylint: disable=invalid-name
# pylint: disable=no-self-use

import numpy as np

# Local imports
from .discrete import DiscreteUniform, Bernoulli
from ..utils.base_test_class import BaseTestClass, execute_tests

class MetropolisDiscreteTestCases(BaseTestClass):
  """ Unit tests for metropolis sampling of discrete distributions """

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

  @unittest.skip
  def test_metropolis_discreteuniform(self):
    """ Tests metropolis sampling from Discrete Uniform distribution """
    self.report('Test metropolis sampling of Discrete Uniform Distribution.')
    lower = -5
    upper = 5
    dist = DiscreteUniform(lower, upper)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('metropolis', self.size, np.array([0.1]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= lower).all() and (samples <= upper).all()
    assert abs(mean - self._compute_mean(samples)) <= self.threshold
    assert abs(var - self._compute_variance(samples)) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

  def test_metropolis_bernoulli(self):
    """ Tests metropolis sampling from Bernoulli distribution """
    self.report('Test metropolis sampling of Bernoulli Distribution.')
    p = 0.6
    dist = Bernoulli(p)
    mean = dist.get_mean()
    var = dist.get_variance()
    samples = dist.draw_samples('binarymetropolis', self.size, np.array([0]))
    mean_r = self._compute_mean(samples)
    var_r = self._compute_variance(samples)
    self._check_sample_sizes(samples)
    assert (samples >= 0).all() and (samples <= 1).all()
    assert abs(mean - self._compute_mean(samples)) <= self.threshold
    assert abs(var - self._compute_variance(samples)) <= self.threshold
    self.report('%s, mean=%0.3f, variance=%0.3f :: test result: mean=%0.3f, '
                'variance=%0.3f'%(str(dist), mean, var, mean_r, var_r), 'test_result')

if __name__ == '__main__':
  execute_tests()
