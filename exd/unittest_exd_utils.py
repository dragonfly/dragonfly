"""
  Unit tests for ed_utils.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import absolute_import
from __future__ import division

# pylint: disable=relative-import

import numpy as np
# Local imports
from exd.exd_utils import latin_hc_indices, latin_hc_sampling
from exd.exd_utils import random_sampling_cts, random_sampling_kmeans_cts
from utils.base_test_class import BaseTestClass, execute_tests


class EDUtilsTestCase(BaseTestClass):
  """ Unit tests for generic functions ed_utils.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.lhs_data = [(1, 10), (2, 5), (4, 10), (10, 100)]

  @classmethod
  def _check_sample_sizes(cls, data, samples):
    """ Data is a tuple of the form (dim, num_samples) ans samples is an ndarray."""
    assert (data[1], data[0]) == samples.shape

  def test_latin_hc_indices(self):
    """ Tests latin hyper-cube index generation. """
    self.report('Test Latin hyper-cube indexing. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_true_sum = data[1] * (data[1] - 1) / 2
      lhs_idxs = latin_hc_indices(data[0], data[1])
      lhs_idx_sums = np.array(lhs_idxs).sum(axis=0)
      assert np.all(lhs_true_sum == lhs_idx_sums)

  def test_latin_hc_sampling(self):
    """ Tests latin hyper-cube sampling. """
    self.report('Test Latin hyper-cube sampling. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_max_sum = float(data[1] + 1)/2
      lhs_min_sum = float(data[1] - 1)/2
      lhs_samples = latin_hc_sampling(data[0], data[1])
      lhs_sample_sums = lhs_samples.sum(axis=0)
      self._check_sample_sizes(data, lhs_samples)
      assert lhs_sample_sums.max() <= lhs_max_sum
      assert lhs_sample_sums.min() >= lhs_min_sum

  def test_random_sampling(self):
    """ Tests random sampling. """
    self.report('Test random sampling.')
    for data in self.lhs_data:
      self._check_sample_sizes(data, random_sampling_cts(data[0], data[1]))

  def test_random_sampling_kmeans(self):
    """ Tests random sampling with k-means. """
    self.report('Test random sampling with k-means.')
    for data in self.lhs_data:
      self._check_sample_sizes(data, random_sampling_kmeans_cts(data[0], data[1]))

  def test_random_sampling_discrete(self):
    """ Tests random sampling from discrete domains. """
    self.report('Test random sampling on a discrete domain.')
    raise NotImplementedError('Implement this!')


if __name__ == '__main__':
  execute_tests()

