"""
  Unit tests for cp_domain_utils.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member

import os
import numpy as np
# Local imports
from . import cp_domain_utils
from ..test_data.park1_3.park1_3 import park1_3
from ..test_data.park1_3.park1_3_mf import park1_3_mf
from ..test_data.park2_4.park2_4 import park2_4
from ..test_data.park2_4.park2_4_mf import park2_4_mf
# from ..test_data.syn_cnn_2.syn_cnn_2 import syn_cnn_2
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.general_utils import get_idxs_from_list_of_lists


class CPDomainUtilsTestCase(BaseTestClass):
  """ A unit test to test ancillary utils in setting up a cartesian product GP. """

  def setUp(self):
    """ Set up. """
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_pardir = os.path.dirname(file_dir)
    self.domain_config_files = [
      test_data_pardir + '/test_data/example_configs/eg01.json',
      test_data_pardir + '/test_data/example_configs/eg02.json',
      test_data_pardir + '/test_data/example_configs/eg03.json',
      test_data_pardir + '/test_data/example_configs/eg01.pb',
      test_data_pardir + '/test_data/example_configs/eg02.pb',
      test_data_pardir + '/test_data/example_configs/eg04.pb',
      test_data_pardir + '/test_data/park2_4/config.json',
      test_data_pardir + '/test_data/park1_3/config.json',
#       test_data_pardir + '/test_data/syn_cnn_2/config.json',
      ]
    self.synthetic_functions = [
      (test_data_pardir + '/test_data/park2_4/config.json', park2_4),
      (test_data_pardir + '/test_data/park1_3/config.json', park1_3),
#       ('test_data/syn_cnn_2/config.json', syn_cnn_2),
      ]
    self.domain_fidel_synthetic_functions = [
      (test_data_pardir + '/test_data/park2_4/config_mf.json', park2_4_mf),
      (test_data_pardir + '/test_data/park1_3/config_mf.json', park1_3_mf),
      ]

  def test_load_domain(self):
    """ Test load domain. """
    # TODO: this only tests for function, not correctness.
    self.report('Testing loading domain from config files - tests only for correctness.')
    for idx, dcf in enumerate(self.domain_config_files):
      self.report('[%d/%d] Testing loading from %s.'%(
                  idx + 1, len(self.domain_config_files), dcf))
      domain, orderings = cp_domain_utils.load_cp_domain_from_config_file(dcf)
      self.report('Domain: %s'%(domain), 'test_result')
      self.report('Ordering: %s'%(orderings), 'test_result')
    self.report('', '')

  def test_load_config_file_without_fidel_space(self):
    """ Tests loading a configuration file - both domain and fidelity space. """
    self.report('Testing loading configuration file with no fidelity space.')
    for dcf in self.domain_config_files:
      configs = cp_domain_utils.load_config_file(dcf)
      assert not hasattr(configs, 'fidel_space')
    self.report('', '')

  @classmethod
  def _test_if_two_elements_are_equal(cls, l1, l2):
    """ Returns True if two lists of objects are equal. """
    for idx in range(len(l1)):
      l1_elem = l1[idx]
      l2_elem = l2[idx]
      assert (hasattr(l1_elem, '__iter__') and hasattr(l2_elem, '__iter__')) or \
             (not hasattr(l1_elem, '__iter__') and not hasattr(l2_elem, '__iter__'))
      if isinstance(l1_elem, np.ndarray) or isinstance(l2_elem, np.ndarray):
        assert np.linalg.norm(np.array(l1_elem) - np.array(l2_elem)) < 1e-3
      else:
        assert l1_elem == l2_elem

  @classmethod
  def _test_if_two_lists_of_objects_are_equal(cls, list_1, list_2):
    """ Returns True if two lists of objects are equal. """
    assert len(list_1) == len(list_2)
    for l1, l2 in zip(list_1, list_2):
      cls._test_if_two_elements_are_equal(l1, l2)

  def test_packing_and_unpacking_points(self):
    """ Tests packing and unpacking. """
    self.report('Testing packing and unpacking points.')
    num_samples = 8
    for idx, dcf in enumerate(self.domain_config_files):
      self.report('[%d/%d] Testing packing/unpacking from %s.'%(
                  idx + 1, len(self.domain_config_files), dcf))
      cp_dom, orderings = cp_domain_utils.load_cp_domain_from_config_file(dcf)
      self.report('Domain: %s'%(cp_dom), 'test_result')
      proc_samples_1 = cp_domain_utils.sample_from_cp_domain(cp_dom, num_samples)
      raw_samples_1 = [cp_domain_utils.get_raw_point_from_processed_point(x, cp_dom,
                         orderings.index_ordering, orderings.dim_ordering)
                       for x in proc_samples_1]
      proc_samples_2 = [cp_domain_utils.get_processed_point_from_raw_point(x, cp_dom,
                          orderings.index_ordering, orderings.dim_ordering)
                        for x in raw_samples_1]
      raw_samples_2 = [cp_domain_utils.get_raw_point_from_processed_point(x, cp_dom,
                         orderings.index_ordering, orderings.dim_ordering)
                       for x in proc_samples_2]
      self._test_if_two_lists_of_objects_are_equal(proc_samples_1, proc_samples_2)
      self._test_if_two_lists_of_objects_are_equal(raw_samples_1, raw_samples_2)
    self.report('', '')

  def test_function_loading_and_processing(self):
    """ Tests creation of the function. """
    self.report('Testing function evaluations before and after packing.')
    num_samples = 25
    for idx, (dcf, raw_func) in enumerate(self.synthetic_functions):
      cp_dom, orderings = cp_domain_utils.load_cp_domain_from_config_file(dcf)
      proc_func = cp_domain_utils.get_processed_func_from_raw_func_for_cp_domain(
                    raw_func, cp_dom, orderings.index_ordering, orderings.dim_ordering)
      # Draw samples
      proc_samples = cp_domain_utils.sample_from_cp_domain(cp_dom, num_samples)
      raw_samples = [cp_domain_utils.get_raw_point_from_processed_point(x, cp_dom,
                         orderings.index_ordering, orderings.dim_ordering)
                       for x in proc_samples]
      # Compute function values
      proc_vals = [proc_func(x) for x in proc_samples]
      raw_vals = [raw_func(x) for x in raw_samples]
      err = np.linalg.norm(np.array(proc_vals) - np.array(raw_vals))
      assert err < 1e-4 * np.linalg.norm(np.array(proc_vals))
      self.report('[%d/%d] Testing function evaluations for %s, Error=%0.5f.'%(
                  idx + 1, len(self.synthetic_functions), dcf, err), 'test_result')
    self.report('', '')

  # Tests for loading configuration files -------------------------------------------
  def test_load_config_file(self):
    """ Tests loading a configuration file - both domain and fidelity space. """
    self.report('Testing loading configuration file.')
    for idx, (dcf, _) in enumerate(self.domain_fidel_synthetic_functions):
      self.report('[%d/%d] Testing loading from %s.'%(
                  idx + 1, len(self.domain_fidel_synthetic_functions), dcf))
      config = cp_domain_utils.load_config_file(dcf)
      self.report('domain: %s'%(config.domain), 'test_result')
      self.report('domain_ordering: %s'%(config.domain_orderings), 'test_result')
      self.report('fidel_space: %s'%(config.fidel_space), 'test_result')
      self.report('fidel_to_opt: %s, (raw: %s)'%(
        config.fidel_to_opt, config.raw_fidel_to_opt), 'test_result')
      self.report('fidel_space_ordering: %s'%(config.fidel_space_orderings),
                  'test_result')
    self.report('', '')

  def test_packing_and_unpacking_with_config(self):
    """ Tests packing and unpacking of points in both fidelity and domain. """
    self.report('Testing packing and unpacking from a configuration file.')
    test_data = self.synthetic_functions[:4] + self.domain_fidel_synthetic_functions
    num_samples = 8
    for idx, (dcf, _) in enumerate(test_data):
      config = cp_domain_utils.load_config_file(dcf)
      has_fidel = hasattr(config, 'fidel_space')
      self.report('[%d/%d] Testing packing/unpacking from %s, has fidel = %d.'%(
        idx + 1, len(test_data), dcf, has_fidel))
      if has_fidel:
        self.report('fidel_space: %s'%(config.fidel_space), 'test_result')
      self.report('domain: %s'%(config.domain), 'test_result')
      proc_samples_1 = cp_domain_utils.sample_from_config_space(config, num_samples)
      raw_samples_1 = [cp_domain_utils.get_raw_from_processed_via_config(x, config)
                       for x in proc_samples_1]
      proc_samples_2 = [cp_domain_utils.get_processed_from_raw_via_config(x, config)
                        for x in raw_samples_1]
      raw_samples_2 = [cp_domain_utils.get_raw_from_processed_via_config(x, config)
                       for x in proc_samples_2]
      if has_fidel:
        fidel_proc_1 = get_idxs_from_list_of_lists(proc_samples_1, 0)
        fidel_proc_2 = get_idxs_from_list_of_lists(proc_samples_2, 0)
        fidel_raw_1 = get_idxs_from_list_of_lists(raw_samples_1, 0)
        fidel_raw_2 = get_idxs_from_list_of_lists(raw_samples_2, 0)
        domain_proc_1 = get_idxs_from_list_of_lists(proc_samples_1, 1)
        domain_proc_2 = get_idxs_from_list_of_lists(proc_samples_2, 1)
        domain_raw_1 = get_idxs_from_list_of_lists(raw_samples_1, 1)
        domain_raw_2 = get_idxs_from_list_of_lists(raw_samples_2, 1)
        # Test if equal
        self._test_if_two_lists_of_objects_are_equal(fidel_proc_1, fidel_proc_2)
        self._test_if_two_lists_of_objects_are_equal(fidel_raw_1, fidel_raw_2)
        self._test_if_two_lists_of_objects_are_equal(domain_proc_1, domain_proc_2)
        self._test_if_two_lists_of_objects_are_equal(domain_raw_1, domain_raw_2)
      else:
        self._test_if_two_lists_of_objects_are_equal(proc_samples_1, proc_samples_2)
        self._test_if_two_lists_of_objects_are_equal(raw_samples_1, raw_samples_2)
    self.report('', '')

  def test_multi_fidelity_function_loading_and_processing_with_config(self):
    """ Tests creation of the function. """
    self.report('Testing multi-fidelity function evals before/after packing with config.')
    num_samples = 25
    for idx, (dcf, raw_func) in enumerate(self.domain_fidel_synthetic_functions):
      config = cp_domain_utils.load_config_file(dcf)
      proc_func = cp_domain_utils.get_processed_func_from_raw_func_via_config(
                    raw_func, config)
      # Draw samples
      proc_samples_1 = cp_domain_utils.sample_from_config_space(config, num_samples)
      raw_samples_1 = [cp_domain_utils.get_raw_from_processed_via_config(x, config)
                       for x in proc_samples_1]
      proc_samples_2 = [cp_domain_utils.get_processed_from_raw_via_config(x, config)
                        for x in raw_samples_1]
      raw_samples_2 = [cp_domain_utils.get_raw_from_processed_via_config(x, config)
                       for x in proc_samples_2]
      # Compute function values
      proc_vals_1 = [proc_func(z, x) for (z, x) in proc_samples_1]
      proc_vals_2 = [proc_func(z, x) for (z, x) in proc_samples_2]
      raw_vals_1 = [raw_func(z, x) for (z, x) in raw_samples_1]
      raw_vals_2 = [raw_func(z, x) for (z, x) in raw_samples_2]
      err_1 = np.linalg.norm(np.array(proc_vals_1) - np.array(raw_vals_1))
      err_2 = np.linalg.norm(np.array(proc_vals_1) - np.array(proc_vals_2))
      err_3 = np.linalg.norm(np.array(proc_vals_1) - np.array(raw_vals_2))
      err = err_1 + err_2 + err_3
      assert err < 1e-4 * np.linalg.norm(np.array(proc_vals_1))
      self.report('[%d/%d] Testing multi-fidelity function evals for %s, Error=%0.5f.'%(
                  idx + 1, len(self.domain_fidel_synthetic_functions), dcf, err),
                  'test_result')
    self.report('', '')


if __name__ == '__main__':
  execute_tests()

