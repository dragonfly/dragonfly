"""
  A unit test for function_caller in function_caller and for synthetic
  functions in utils.euclidean_synthetic_functions.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

import numpy as np
# Local imports
import utils.euclidean_synthetic_functions as esf
from utils.base_test_class import BaseTestClass, execute_tests
from utils.ancillary_utils import get_list_of_floats_as_str

_TOL = 1e-5


class EuclideanSyntheticFunctionTestCase(BaseTestClass):
  """ Unit test for synthetic function. """

  def setUp(self):
    """ Prior set up. """
    self.test_function_data = [('hartmann', 3, None, 'no_noise', None),
                               ('hartmann3', None, None, 'no_noise', None),
                               ('hartmann6', None, None, 'no_noise', None),
                               ('branin', None, None, 'no_noise', None),
                               ('borehole', None, None, 'no_noise', None),
                               # With noise
                               ('hartmann6', None, None, 'gauss', 0.1),
                               ('branin', None, None, 'gauss', 0.1),
                               ('borehole', None, None, 'gauss', 10.0),
                               # Multi-fidelity with and without noise
                               ('hartmann3', None, 2, 'no_noise', None),
                               ('branin', None, 3, 'no_noise', None),
                               ('hartmann6', None, 4, 'gauss', 0.1),
                               ('borehole', None, 1, 'gauss', 10.0),
                              ]
    self.num_test_points = 100

  @classmethod
  def _get_test_vals_from_func_caller(cls, func_caller, test_points):
    """ Gets the test points. """
    test_rets = [func_caller.eval_single(elem) for elem in test_points]
    test_vals = [ret[0] for ret in test_rets]
    test_qinfos = [ret[1] for ret in test_rets]
    return test_vals, test_qinfos, test_rets

  @classmethod
  def _get_test_vals_from_mf_func_caller(cls, func_caller, test_points, test_fidels):
    """ Gets the test points. """
    test_rets = [func_caller.eval_at_fidel_single(fidel, point) for \
                 fidel, point in zip(test_fidels, test_points)]
    test_vals = [ret[0] for ret in test_rets]
    test_qinfos = [ret[1] for ret in test_rets]
    return test_vals, test_qinfos, test_rets

  def _test_for_max_val(self, func_caller):
    """ Tests that the optimum value is larger than the test points. """
    test_points = np.random.random((self.num_test_points, func_caller.domain.dim))
    fc_test_vals, _, _ = self._get_test_vals_from_func_caller(func_caller, test_points)
    assert np.all(np.array(fc_test_vals) <= func_caller.maxval)

  @classmethod
  def _test_for_test_err(cls, test_vals_1, test_vals_2, noise_scale):
    """ A single function to do the testing. """
    test_err = np.linalg.norm(np.array(test_vals_1) - np.array(test_vals_2)) / \
               np.sqrt(len(test_vals_1))
    assert test_err <= _TOL + 2 * noise_scale
    return test_err

  def _test_for_func_vals(self, func_caller, unnorm_func_caller):
    """ Tests that the function values correspond to those given. The unit test
        test_all_synthetic_functions calls this for each of the test functions in
        a loop.
    """
    # In this unit test, fc_ denotes for quantities corresponding to the (normalised)
    # func_caller. ufc_ denotes quantities corresponding to the unnorm_func_caller.
    # Test with func_caller and func
    test_points = np.random.random((self.num_test_points, func_caller.domain.dim))
    raw_test_points = func_caller.get_raw_domain_coords(test_points)
    if not func_caller.is_mf():
      raw_test_vals = [func_caller.func(elem) for elem in raw_test_points]
    else:
      test_fidels = np.random.random((self.num_test_points, func_caller.fidel_space.dim))
      raw_test_fidels = func_caller.get_raw_fidel_coords(test_fidels)
      raw_test_vals = [func_caller.func(rfidel, rpoint) for
                       (rfidel, rpoint) in zip(raw_test_fidels, raw_test_points)]
    # First the normalised function caller
    fc_noise_scale = 0.0 if func_caller.noise_scale is None else func_caller.noise_scale
    if not func_caller.is_mf():
      fc_test_vals, _, _ = self._get_test_vals_from_func_caller(func_caller, test_points)
    else:
      fc_test_vals, _, _ = self._get_test_vals_from_mf_func_caller(func_caller,
                             test_points, test_fidels)
    fc_test_err = self._test_for_test_err(raw_test_vals, fc_test_vals, fc_noise_scale)
    # Test with unnormalised function caller
    ufc_noise_scale = 0.0 if func_caller.noise_scale is None else func_caller.noise_scale
    if not func_caller.is_mf():
      ufc_test_vals, _, _ = self._get_test_vals_from_func_caller(unnorm_func_caller,
                                                                 raw_test_points)
    else:
      ufc_test_vals, _, _ = self._get_test_vals_from_mf_func_caller(unnorm_func_caller,
                              raw_test_points, raw_test_fidels)
    ufc_test_err = self._test_for_test_err(raw_test_vals, ufc_test_vals, ufc_noise_scale)
    # Test for normalised and unnormalised callers
    fc_ufc_noise_scale = fc_noise_scale + ufc_noise_scale
    fc_ufc_err = self._test_for_test_err(fc_test_vals, ufc_test_vals, fc_ufc_noise_scale)
    return fc_test_err, ufc_test_err, fc_ufc_err

  def _test_variation_along_fidel_dim(self, func_caller, unnorm_func_caller,
                                      fidel_dim, test_point=None):
    """ Tests and prints variation along the a fidelity dim. """
    _grid_size = 10
    _rem_coords_pre = np.ones((_grid_size, fidel_dim))
    _fidel_dim_coords = np.reshape(np.linspace(0, 1, _grid_size), (_grid_size, 1))
    _rem_coords_post = np.ones((_grid_size, func_caller.fidel_space.dim - fidel_dim - 1))
    fidel_test_grid = np.hstack((_rem_coords_pre, _fidel_dim_coords, _rem_coords_post))
    raw_fidel_test_grid = func_caller.get_raw_fidel_coords(fidel_test_grid)
    test_point = test_point if test_point is not None else \
                 np.random.random((func_caller.domain.dim,))
    raw_test_point = func_caller.get_raw_domain_coords(test_point)
    test_vals_at_grid = [func_caller.eval_at_fidel_single(fidel, test_point)[0] for
                         fidel in fidel_test_grid]
    ufc_test_vals_at_grid = [
      unnorm_func_caller.eval_at_fidel_single(raw_fidel, raw_test_point)[0]
      for raw_fidel in raw_fidel_test_grid]
    test_vals_at_grid_str = get_list_of_floats_as_str(test_vals_at_grid)
    ufc_test_vals_at_grid_str = get_list_of_floats_as_str(ufc_test_vals_at_grid)
    test_point_str = get_list_of_floats_as_str(test_point)
    raw_test_point_str = get_list_of_floats_as_str(raw_test_point)
    self.report('fidel values (normalised func_caller) at x=%s: (fidel_dim=%d) %s'%(
                test_point_str, fidel_dim, test_vals_at_grid_str), 'test_result')
    self.report('fidel values (unnorm_func_caller) at x=%s:  (fidel_dim=%d) %s'%(
                raw_test_point_str, fidel_dim, ufc_test_vals_at_grid_str), 'test_result')

  def test_all_synthetic_functions(self):
    """ Tests all synthetic functions in a loop. """
    for idx, (func_name, domain_dim, fidel_dim, noise_type, noise_scale) in \
      enumerate(self.test_function_data):
      sf_or_mf = 'sf' if fidel_dim is None else 'mf'
      self.report(('Testing %d/%d: %s(%s), domain_dim:%s, fidel_dim:%s, ' +
                   'noise(%s, %s).')%(idx+1, len(self.test_function_data),
                  func_name, sf_or_mf, domain_dim, fidel_dim,
                  noise_type, noise_scale))
      # get the function
      func_caller = esf.get_syn_func_caller(func_name, domain_dim, fidel_dim,
                                            noise_type, noise_scale,
                                            to_normalise_domain=True)
      unnorm_func_caller = esf.get_syn_func_caller(func_name, domain_dim, fidel_dim,
                                                   noise_type, noise_scale,
                                                   to_normalise_domain=False)
      # Test for the maximum values
      if not func_caller.is_noisy():
        self._test_for_max_val(func_caller)
      # Test for the function values and print out result
      fc_err, ufc_err, fc_ufc_err = \
        self._test_for_func_vals(func_caller, unnorm_func_caller)
      self.report(('normalised_err: %0.5f, unnormalised_err: %0.5f., ' +
                   'norm-unnorm_err: %0.5f')%(fc_err, ufc_err, fc_ufc_err), 'test_result')
      # Print variation along a fidelity dimension
      if func_caller.is_mf() and not func_caller.is_noisy():
        self._test_variation_along_fidel_dim(func_caller, unnorm_func_caller, 0)


if __name__ == '__main__':
  execute_tests()

