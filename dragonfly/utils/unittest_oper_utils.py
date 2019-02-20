"""
  Unit tests for optimers.
  -- kandasamy@cs.cmu.edu
"""
from __future__ import absolute_import

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import os
import time
from argparse import Namespace
import numpy as np
# Local
from .ancillary_utils import is_nondecreasing
from .base_test_class import BaseTestClass, execute_tests
from .euclidean_synthetic_functions import get_mf_hartmann_function_data
from . import oper_utils

# TODO: Write unit tests for Optimal Transport


class OptimisersTestCase(BaseTestClass):
  """Unit test class for general utilities. """

  def __init__(self, *args, **kwargs):
    super(OptimisersTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Sets up attributes. """
    self.problems = []
    self.max_evals = 20000 #1e4
    # First problem
    obj = lambda x: np.dot(x-1, x)
    dim = 4
    min_pt = np.array([0.5] * dim)
    max_pt = np.array([-1] * dim)
    self.problems.append(self._get_test_case_problem_instance(obj, [-1] * dim, \
                                               [1] * dim, min_pt, max_pt, '4D-quadratic'))
    # Second problem
    obj = lambda x: np.exp(-np.dot(x-0.5, x))
    dim = 2
    min_pt = np.array([-1] * dim)
    max_pt = np.array([0.25] * dim)
    self.problems.append(self._get_test_case_problem_instance(obj, [-1] * dim, \
                                                [1] * dim, min_pt, max_pt, '2D-gaussian'))
    # Third problem - use Hartmann6
    hartmann_data = get_mf_hartmann_function_data(0, 6)
    obj = hartmann_data[1]
    max_pt = hartmann_data[2]
    min_pt = None
    domain_bounds = np.array(hartmann_data[-1])
    self.problems.append(self._get_test_case_problem_instance(obj, domain_bounds[:, 0],
      domain_bounds[:, 1], min_pt, max_pt, 'Hartmann6'))

  @classmethod
  def _get_test_case_problem_instance(cls, obj, lb, ub, min_pt, max_pt, descr=''):
    """ A wrapper which returns a problem instance as a list. """
    min_val = float(obj(min_pt)) if min_pt is not None else None
    max_val = float(obj(max_pt)) if max_pt is not None else None
    bounds = np.vstack((lb, ub)).T
    problem_inst = Namespace(obj=obj, dim=len(lb), lb=lb, ub=ub, bounds=bounds,
                             min_pt=min_pt, max_pt=max_pt,
                             min_val=min_val, max_val=max_val,
                             descr=descr)
    return problem_inst

  def _test_optimise_method(self, minimise_method, maximise_method, test_success=True):
    """ Tests an optimiser method. """
    num_min_successes = 0
    num_max_successes = 0
    for prob in self.problems:
      # First the minimimum
      if prob.min_val is not None:
        min_val_soln, min_pt_soln, _ = minimise_method(prob.obj, prob.bounds,
                                                       self.max_evals)
        val_diff = abs(prob.min_val - min_val_soln)
        point_diff = np.linalg.norm(prob.min_pt - min_pt_soln)
        self.report(prob.descr +
          '(min):: true-val: %0.4f, soln: %0.4f, diff: %0.4f.'%(prob.min_val,
             min_val_soln, val_diff), 'test_result')
        self.report(prob.descr +
          '(min):: true-pt: %s, soln: %s, diff: %0.4f.'%(prob.min_pt, min_pt_soln,
             point_diff), 'test_result')
        min_is_successful = val_diff < 1e-3 and point_diff < 1e-3 * prob.dim
        num_min_successes += min_is_successful
      else:
        num_min_successes += 1
      # Now the maximum
      if prob.max_val is not None:
        max_val_soln, max_pt_soln, _ = maximise_method(prob.obj, prob.bounds,
                                                       self.max_evals)
        val_diff = abs(prob.max_val - max_val_soln)
        point_diff = np.linalg.norm(prob.max_pt - max_pt_soln)
        self.report(prob.descr +
          '(max):: true-val: %0.4f, soln: %0.4f, diff: %0.4f.'%(prob.max_val,
             max_val_soln, val_diff), 'test_result')
        self.report(prob.descr +
          '(max):: true-pt: %s, soln: %s, diff: %0.4f.'%(prob.max_pt, max_pt_soln,
             point_diff), 'test_result')
        max_is_successful = val_diff < 1e-3 and point_diff < 1e-3 * prob.dim
        num_max_successes += max_is_successful
      else:
        num_max_successes += max_is_successful
    # Check if successful
    if test_success:
      assert num_min_successes == len(self.problems)
      assert num_max_successes == len(self.problems)

  def test_random_optimise(self):
    """ Test direct optmisation."""
    self.report('Rand optimise:')
    random_minimise = lambda obj, bounds, max_evals: oper_utils.random_minimise(
                                                obj, bounds, max_evals, False, False)
    random_maximise = lambda obj, bounds, max_evals: oper_utils.random_maximise(
                                                obj, bounds, max_evals, False, False)
    self._test_optimise_method(random_minimise, random_maximise, False)

  def test_direct(self):
    """ Test direct optmisation."""
    self.report('DiRect minimise and maximise:')
    self._test_optimise_method(oper_utils.direct_ft_minimise,
                               oper_utils.direct_ft_maximise)

  def test_direct_times(self):
    """ Tests the running time of the package with and without file writing. """
    self.report('DiRect running times')
    log_file_names = ['', 'test_log']
    for prob in self.problems:
      clock_times = []
      real_times = []
      for log_file_name in log_file_names:
        start_clock = time.clock()
        start_real_time = time.time()
        _, _, _ = oper_utils.direct_ft_maximise(prob.obj, prob.bounds, self.max_evals,
                                                log_file_name=log_file_name)
        clock_time = time.clock() - start_clock
        real_time = time.time() - start_real_time
        clock_times.append(clock_time)
        real_times.append(real_time)
        if log_file_name:
          try:
            os.remove(log_file_name)
          except OSError:
            pass
      # Print results out
      result_str = ', '.join(['file: \'%s\': clk=%0.4f, real=%0.4f, #evals=%d'%(
        log_file_names[i], clock_times[i], real_times[i], self.max_evals) \
        for i in range(len(log_file_names))])
      self.report('%s:: %s'%(prob.descr, result_str), 'test_result')

  def test_direct_with_history(self):
    """ Tests direct with history. """
    self.report('DiRect with history.')
    for prob in self.problems:
      min_val_soln, _, history = oper_utils.direct_ft_maximise(prob.obj, prob.bounds, \
                                                      self.max_evals, return_history=True)
      if history is not None:
        assert is_nondecreasing(history.curr_opt_vals)
        assert np.abs(min_val_soln - history.curr_opt_vals[-1]) < 1e-4

  def test_pdoo(self):
    """ Test PDOO optmisation."""
    self.report('PDOO minimise and maximise:')
    self._test_optimise_method(oper_utils.pdoo_minimise, oper_utils.pdoo_maximise)

  def test_pdoo_times(self):
    """ Tests the running time of the package with and without file writing. """
    self.report('PDOO running times.')
    for prob in self.problems:
      start_clock = time.clock()
      start_real_time = time.time()
      _, _, _ = oper_utils.pdoo_maximise(prob.obj, prob.bounds, self.max_evals)
      clock_time = time.clock() - start_clock
      real_time = time.time() - start_real_time
      self.report('clk=%0.4f, real=%0.4f, #evals=%d'%(
                  clock_time, real_time, self.max_evals), 'test_result')


class SamplersTestCase(BaseTestClass):
  """Unit test class for sampling utilities. """

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
      lhs_idxs = oper_utils.latin_hc_indices(data[0], data[1])
      lhs_idx_sums = np.array(lhs_idxs).sum(axis=0)
      assert np.all(lhs_true_sum == lhs_idx_sums)

  def test_latin_hc_sampling(self):
    """ Tests latin hyper-cube sampling. """
    self.report('Test Latin hyper-cube sampling. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_max_sum = float(data[1] + 1)/2
      lhs_min_sum = float(data[1] - 1)/2
      lhs_samples = oper_utils.latin_hc_sampling(data[0], data[1])
      lhs_sample_sums = lhs_samples.sum(axis=0)
      self._check_sample_sizes(data, lhs_samples)
      assert lhs_sample_sums.max() <= lhs_max_sum
      assert lhs_sample_sums.min() >= lhs_min_sum



if __name__ == '__main__':
  execute_tests()

