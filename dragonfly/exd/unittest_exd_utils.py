"""
  Unit tests for ed_utils.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import absolute_import
from __future__ import division

# pylint: disable=relative-import

# Local imports
from .exd_utils import random_sampling_cts, random_sampling_kmeans_cts
from ..utils.base_test_class import BaseTestClass, execute_tests


class EDUtilsTestCase(BaseTestClass):
  """ Unit tests for generic functions ed_utils.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.lhs_data = [(1, 10), (2, 5), (4, 10), (10, 100)]

  @classmethod
  def _check_sample_sizes(cls, data, samples):
    """ Data is a tuple of the form (dim, num_samples) ans samples is an ndarray."""
    assert (data[1], data[0]) == samples.shape

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


if __name__ == '__main__':
  execute_tests()

