"""
  Unit tests for continuous distributions.
  -- kvysyara@andrew.cmu.edu
"""
from __future__ import absolute_import

# pylint: disable=invalid-name
# pylint: disable=no-self-use

import numpy as np

# Local imports
from .continuous import Normal, Exponential, ContinuousUniform
from .product import JointDistribution
from ..utils.base_test_class import BaseTestClass, execute_tests

class JointDistributionTestCase(BaseTestClass):
  """ Unit tests for joint distribution class """

  def setUp(self):
    """ Sets up unit tests. """
    self.size = 1000000
    self.threshold = 0.1

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

  def test_rand_sampling(self):
    """ Tests random sampling from Joint distribution """
    self.report('Test random sampling of Joint Distribution.')
    normal_rv = Normal(0, 1)
    exp_rv = Exponential(2)
    uniform_rv = ContinuousUniform(-5, 5)
    dist = JointDistribution([normal_rv, exp_rv, uniform_rv])
    samples = dist.draw_samples('random', self.size)
    for i, rv in enumerate(dist.get_rv_s()):
      self._check_sample_sizes(samples[i])
      assert abs(rv.get_mean() - self._compute_mean(samples[i])) <= self.threshold
      assert abs(rv.get_variance() - self._compute_variance(samples[i])) <= self.threshold

if __name__ == '__main__':
  execute_tests()

