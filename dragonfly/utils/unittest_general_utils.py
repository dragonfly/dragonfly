"""
  Test cases for functions in general_utils.py
  -- kandasamy@cs.cmu.edu
"""
from __future__ import absolute_import
from __future__ import division

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np
from . import general_utils
from .base_test_class import BaseTestClass, execute_tests


class GeneralUtilsTestCase(BaseTestClass):
  """Unit test class for general utilities. """

  def __init__(self, *args, **kwargs):
    super(GeneralUtilsTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Sets up attributes. """
    # For dist squared
    self.X1 = np.array([[1, 2, 3], [1, 2, 4], [2, 3, 4.5]])
    self.X2 = np.array([[1, 2, 4], [1, 2, 5], [2, 3, 5]])
    self.true_dist_sq = np.array([[1, 4, 6], [0, 1, 3], [2.25, 2.25, 0.25]])

  def test_dist_squared(self):
    """ Tests the squared distance function. """
    self.report('dist_squared')
    comp_dist_sq = general_utils.dist_squared(self.X1, self.X2)
    assert (self.true_dist_sq == comp_dist_sq).all()

  def test_mapping_to_cube_and_bound(self):
    """ Test map_to_cube and map_to_bounds. """
    self.report('map_to_cube and map_to_bounds')
    bounds = np.array([[1, 3], [2, 4], [5, 6]])
    x = np.array([1.7, 3.1, 5.5])
    X = np.array([[1.7, 3.1, 5.5], [2.1, 2.9, 5.0]])
    y = np.array([0.35, 0.55, 0.5])
    Y = np.array([[0.35, 0.55, 0.5], [0.55, 0.45, 0]])
    # Map to cube
    y_ = general_utils.map_to_cube(x, bounds)
    Y_ = general_utils.map_to_cube(X, bounds)
    # Map to Bounds
    x_ = general_utils.map_to_bounds(y, bounds)
    X_ = general_utils.map_to_bounds(Y, bounds)
    # Check if correct.
    assert np.linalg.norm(y - y_) < 1e-5
    assert np.linalg.norm(Y - Y_) < 1e-5
    assert np.linalg.norm(x - x_) < 1e-5
    assert np.linalg.norm(X - X_) < 1e-5

  def test_compute_average_sq_prediction_error(self):
    """ Tests compute_average_sq_prediction_error. """
    self.report('compute_average_sq_prediction_error')
    Y1 = [0, 1, 2]
    Y2 = [2, 0, 1]
    res = general_utils.compute_average_sq_prediction_error(Y1, Y2)
    assert np.abs(res - 2.0) < 1e-5

  def test_stable_cholesky(self):
    """ Tests for stable cholesky. """
    self.report('stable_cholesky')
    M = np.random.normal(size=(5, 5))
    M = M.dot(M.T)
    L = general_utils.stable_cholesky(M)
    assert np.linalg.norm(L.dot(L.T) - M) < 1e-5

  def test_draw_gaussian_samples(self):
    """ Tests for draw gaussian samples. """
    self.report('draw_gaussian_samples. Probabilistic test, could fail at times')
    num_samples = 10000
    num_pts = 3
    mu = list(range(num_pts))
    K = np.random.normal(size=(num_pts, num_pts))
    K = K.dot(K.T)
    samples = general_utils.draw_gaussian_samples(num_samples, mu, K)
    sample_mean = samples.mean(axis=0)
    sample_centralised = samples - sample_mean
    sample_covar = sample_centralised.T.dot(sample_centralised) / num_samples
    mean_tol = 4 * np.linalg.norm(mu) / np.sqrt(num_samples)
    covar_tol = 4 * np.linalg.norm(K) / np.sqrt(num_samples)
    mean_err = np.linalg.norm(mu - sample_mean)
    covar_err = np.linalg.norm(K - sample_covar)
    self.report('Mean error (tol): ' + str(mean_err) + ' (' + str(mean_tol) + ')',
                'test_result')
    self.report('Cov error (tol): ' + str(covar_err) + ' (' + str(covar_tol) + ')',
                'test_result')
    assert mean_err < mean_tol
    assert covar_err < covar_tol

  def test_ordering_and_reordering(self):
    """ Tests for ordering and reordering of lists. """
    self.report('Testing ordering and reordering of lists.')
    # Test 1
    orig_list = ['k', 'i', 'r', 't', 'h']
    ordering = [3, 1, 0, 4, 2]
    reordered_list = general_utils.reorder_list(orig_list, ordering)
    rereordered_list = general_utils.get_original_order_from_reordered_list(
                         reordered_list, ordering)
    assert reordered_list == ['t', 'i', 'k', 'h', 'r']
    assert rereordered_list == orig_list
    # Test 2
    N = 20
    orig_list = list(np.random.random(N,))
    ordering = np.random.permutation(N)
    reordered_list = general_utils.reorder_list(orig_list, ordering)
    rereordered_list = general_utils.get_original_order_from_reordered_list(
                         reordered_list, ordering)
    assert rereordered_list == orig_list

  def test_list_flattening(self):
    """ Tests list flattening. """
    self.report('Testing list flattening.')
    # Test 1
    test_lists = [[4, 5, 1],
                  [4, 5, 6, ['k', None], 5]]
    test_results = [[4, 5, 1],
                    [4, 5, 6, 'k', None, 5]]
    for tl, tr in zip(test_lists, test_results):
      result = general_utils.flatten_list_of_objects_and_lists(tl)
      assert tr == result
    # Test 2
    test_lists.extend([['q', 'we', [7, ['3', str, int], 9], 'k']])
    test_results.extend([['q', 'we', 7, '3', str, int, 9, 'k']])
    for tl, tr in zip(test_lists, test_results):
      result = general_utils.flatten_nested_lists(tl)
      assert tr == result

  def test_pairwise_hamming_kernel(self):
    """ Tests pairwise hamming distances. """
    self.report('Testing hamming distance.')
    # Test 1
    test_data = [# data 1
                ([[4, 5], ['kky', 5], ['s', None]],
                 [[2, 5], ['s', None], ['kky', None], ['s', 1]],
                 [0.4, 0.6],
                 [[1.0, 0.6, 0], [0.6, 1.0, 0], [0, 0, 1.0]],
                 [[1, 0, 0, 0], [0, 1, 0.6, 0.4], [0, 0.6, 1, 0], [0, 0.4, 0, 1]],
                 [[0.6, 0, 0, 0], [0.6, 0, 0.4, 0], [0, 1, 0.6, 0.4]]),
                 # data 2
                ([[4, 5, 6], ['kky', 5, 'cat']],
                 [[2, 5, 6], ['s', None, 'cat'], ['kky', None, 6]],
                 None,
                 [[1.0, 1/3.0], [1/3.0, 1]],
                 [[1.0, 0, 1/3.0], [0, 1, 1/3.0], [1/3.0, 1/3.0, 1]],
                 [[2/3.0, 0, 1/3.0], [1/3.0, 1/3.0, 1/3.0]]),
                ]
    for td in test_data:
      res_00 = np.array(td[-3])
      res_11 = np.array(td[-2])
      res_01 = np.array(td[-1])
      res_10 = res_01.T
      kern_00 = general_utils.pairwise_hamming_kernel(td[0], td[0], td[2])
      kern_11 = general_utils.pairwise_hamming_kernel(td[1], td[1], td[2])
      kern_01 = general_utils.pairwise_hamming_kernel(td[0], td[1], td[2])
      kern_10 = general_utils.pairwise_hamming_kernel(td[1], td[0], td[2])
#       print res_00
#       print kern_00
#       print res_11
#       print kern_11
#       print res_01
#       print kern_01
#       print res_10
#       print kern_10
      assert np.linalg.norm(res_00 - kern_00) < 1e-5
      assert np.linalg.norm(res_11 - kern_11) < 1e-5
      assert np.linalg.norm(res_01 - kern_01) < 1e-5
      assert np.linalg.norm(res_10 - kern_10) < 1e-5


if __name__ == '__main__':
  execute_tests()

